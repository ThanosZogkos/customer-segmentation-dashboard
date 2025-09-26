import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentation:
   
   # Handles multiple clustering algorithms for customer segmentation analysis.
   
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.hierarchical_model = None
        self.scaled_features = None
        self.feature_names = None
        self.last_algorithm = None
    
    def find_optimal_clusters(self, data, max_clusters=10):
        
        #Find optimal number of clusters using elbow method and silhouette analysis.
        
        try:
            # Scale the features
            scaled_data = self.scaler.fit_transform(data)
            self.scaled_features = scaled_data
            self.feature_names = data.columns.tolist()
            
            inertias = []
            silhouette_scores = []
            
            # Calculate metrics for different cluster numbers
            for k in range(1, max_clusters + 1):
                kmeans = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    random_state=self.random_state,
                    n_init='auto',
                    max_iter=300
                )
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
                
                # Calculate silhouette score for k > 1
                if k > 1:
                    score = silhouette_score(scaled_data, kmeans.labels_)
                    silhouette_scores.append(score)
            
            st.success(f"Completed clustering analysis for {max_clusters} cluster configurations")
            return inertias, silhouette_scores
            
        except Exception as e:
            st.error(f"Error in optimal cluster analysis: {str(e)}")
            raise e
    
    def get_optimal_clusters_elbow(self, inertias):
        
        #Determine optimal number of clusters using elbow method.
        
        try:
            # Calculate the rate of change in inertias
            if len(inertias) < 3:
                return 3  # Default fallback
            
            # Calculate second derivatives to find the elbow
            deltas = np.diff(inertias)
            second_deltas = np.diff(deltas)
            
            # Find the point where the rate of change starts to level off
            # Look for the maximum second derivative (biggest change in slope)
            if len(second_deltas) > 0:
                elbow_index = np.argmax(second_deltas) + 2  # +2 because of double diff
                optimal_k = min(max(int(elbow_index), 3), len(inertias))  # Ensure reasonable range
            else:
                optimal_k = 4  # Default fallback
            
            st.info(f"Elbow method suggests {optimal_k} clusters as optimal")
            return optimal_k
            
        except Exception as e:
            st.warning(f"Could not determine optimal clusters: {str(e)}")
            return 4  # Safe fallback
    
    def perform_clustering(self, data, feature_columns, n_clusters):
        
       # Perform K-Means++ clustering on the customer data.
        
        try:
            # Prepare features for clustering
            features = data[feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # Perform K-Means++ clustering
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                init='k-means++',
                random_state=self.random_state,
                n_init='auto',
                max_iter=300
            )
            
            cluster_labels = self.kmeans_model.fit_predict(scaled_features)
            
            # Add cluster labels to the original data
            clustered_df = data.copy()
            clustered_df['Cluster'] = cluster_labels
            
            # Get cluster centers in original scale
            cluster_centers = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
            
            # Calculate clustering quality metrics
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            inertia = self.kmeans_model.inertia_
            
            st.success(f"""
            **Clustering Complete!**
            - Number of clusters: {n_clusters}
            - Silhouette score: {silhouette_avg:.3f}
            - Inertia: {inertia:.2f}
            """)
            
            # Display cluster summary
            self.display_cluster_summary(clustered_df, cluster_centers, feature_columns)
            
            return clustered_df, cluster_centers
            
        except Exception as e:
            st.error(f"Error performing clustering: {str(e)}")
            raise e
    
    def display_cluster_summary(self, clustered_df, cluster_centers, feature_columns):
       
        #Display summary of clustering results.
       
        try:
            st.subheader("Cluster Summary")
            
            # Create cluster summary table
            cluster_summary = []
            for i in range(len(cluster_centers)):
                cluster_data = clustered_df[clustered_df['Cluster'] == i]
                
                summary = {
                    'Cluster': i,
                    'Size': len(cluster_data),
                    'Percentage': (len(cluster_data) / len(clustered_df)) * 100
                }
                
                # Add center coordinates
                for j, feature in enumerate(feature_columns):
                    summary[f'{feature}_Center'] = cluster_centers[i][j]
                
                cluster_summary.append(summary)
            
            summary_df = pd.DataFrame(cluster_summary)
            st.dataframe(summary_df.round(2), use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not display cluster summary: {str(e)}")
    
    def predict_cluster(self, new_data, feature_columns):
        
       # Predict cluster for new customer data.
        
        try:
            if self.kmeans_model is None:
                raise ValueError("Model not trained. Please perform clustering first.")
            
            features = new_data[feature_columns].values
            scaled_features = self.scaler.transform(features)
            predictions = self.kmeans_model.predict(scaled_features)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error predicting clusters: {str(e)}")
            raise e
    
    def calculate_cluster_distances(self, data, feature_columns):
       
        #Calculate distance of each point from its cluster center.
        
        try:
            if self.kmeans_model is None:
                raise ValueError("Model not trained. Please perform clustering first.")
            
            features = data[feature_columns].values
            scaled_features = self.scaler.transform(features)
            
            # Calculate distances to assigned cluster centers
            distances = []
            for i, point in enumerate(scaled_features):
                cluster_label = data.iloc[i]['Cluster']
                center = self.kmeans_model.cluster_centers_[cluster_label]
                distance = np.linalg.norm(point - center)
                distances.append(distance)
            
            return np.array(distances)
            
        except Exception as e:
            st.warning(f"Could not calculate cluster distances: {str(e)}")
            return np.array([])
    
    def get_cluster_characteristics(self, clustered_df, feature_columns):
        
        #Get detailed characteristics of each cluster.
        
        try:
            characteristics = {}
            
            for cluster_id in sorted(clustered_df['Cluster'].unique()):
                cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
                
                char = {
                    'size': len(cluster_data),
                    'percentage': (len(cluster_data) / len(clustered_df)) * 100,
                    'demographics': {},
                    'behavior': {}
                }
                
                # Demographic characteristics
                char['demographics']['avg_age'] = cluster_data['Age'].mean()
                char['demographics']['age_std'] = cluster_data['Age'].std()
                char['demographics']['gender_dist'] = cluster_data['Gender'].value_counts().to_dict()
                
                # Behavioral characteristics
                char['behavior']['avg_income'] = cluster_data['Annual Income (k$)'].mean()
                char['behavior']['income_std'] = cluster_data['Annual Income (k$)'].std()
                char['behavior']['avg_spending'] = cluster_data['Spending Score (1-100)'].mean()
                char['behavior']['spending_std'] = cluster_data['Spending Score (1-100)'].std()
                
                # Feature-specific stats
                for feature in feature_columns:
                    char[f'{feature}_stats'] = {
                        'mean': cluster_data[feature].mean(),
                        'std': cluster_data[feature].std(),
                        'min': cluster_data[feature].min(),
                        'max': cluster_data[feature].max(),
                        'median': cluster_data[feature].median()
                    }
                
                characteristics[cluster_id] = char
            
            return characteristics
            
        except Exception as e:
            st.warning(f"Could not calculate cluster characteristics: {str(e)}")
            return {}
    
    def perform_dbscan_clustering(self, data, feature_columns, eps=0.5, min_samples=5):
        
       # Perform DBSCAN clustering on the customer data.

        try:
            # Prepare features for clustering
            features = data[feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # Perform DBSCAN clustering
            self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = self.dbscan_model.fit_predict(scaled_features)
            
            # Add cluster labels to the original data
            clustered_df = data.copy()
            clustered_df['Cluster'] = cluster_labels
            
            # Calculate number of clusters (excluding noise points labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = (cluster_labels == -1).sum()
            
            # Calculate cluster centers (excluding noise points)
            cluster_centers = []
            unique_labels = set(cluster_labels)
            # Remove noise label if present
            unique_labels = {label for label in unique_labels if label != -1}
            
            for label in sorted([int(l) for l in unique_labels]):
                mask = cluster_labels == label
                center = scaled_features[mask].mean(axis=0)
                cluster_centers.append(self.scaler.inverse_transform(center.reshape(1, -1))[0])
            
            cluster_centers = np.array(cluster_centers) if cluster_centers else np.array([])
            
            # Calculate clustering quality metrics (only for non-noise points)
            if n_clusters > 1:
                non_noise_mask = cluster_labels != -1
                non_noise_features = scaled_features[non_noise_mask]
                non_noise_labels = cluster_labels[non_noise_mask]
                
                if len(set(non_noise_labels)) > 1:
                    silhouette_avg = silhouette_score(non_noise_features, non_noise_labels)
                    calinski_score = calinski_harabasz_score(non_noise_features, non_noise_labels)
                    davies_bouldin = davies_bouldin_score(non_noise_features, non_noise_labels)
                else:
                    silhouette_avg = calinski_score = davies_bouldin = 0
            else:
                silhouette_avg = calinski_score = davies_bouldin = 0
            
            self.last_algorithm = 'DBSCAN'
            
            st.success(f"""
             **DBSCAN Clustering Complete!**
            - Number of clusters: {n_clusters}
            - Noise points: {n_noise}
            - Silhouette score: {silhouette_avg:.3f}
            - Calinski-Harabasz score: {calinski_score:.2f}
            - Davies-Bouldin score: {davies_bouldin:.3f}
            """)
            
            return clustered_df, cluster_centers, n_clusters
            
        except Exception as e:
            st.error(f"Error performing DBSCAN clustering: {str(e)}")
            raise e
    
    def perform_hierarchical_clustering(self, data, feature_columns, n_clusters, linkage_method='ward'):
        """
        Perform Hierarchical clustering on the customer data.
        
        Args:
            data (pd.DataFrame): Customer data
            feature_columns (list): Columns to use for clustering
            n_clusters (int): Number of clusters
            linkage_method (str): Linkage method ('ward', 'complete', 'average', 'single')
            
        Returns:
            tuple: (clustered_dataframe, cluster_centers, linkage_matrix)
        """
        try:
            # Prepare features for clustering
            features = data[feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # Perform Hierarchical clustering
            self.hierarchical_model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage_method
            )
            cluster_labels = self.hierarchical_model.fit_predict(scaled_features)
            
            # Add cluster labels to the original data
            clustered_df = data.copy()
            clustered_df['Cluster'] = cluster_labels
            
            # Calculate cluster centers
            cluster_centers = []
            for i in range(n_clusters):
                mask = cluster_labels == i
                if mask.sum() > 0:
                    center = scaled_features[mask].mean(axis=0)
                    cluster_centers.append(self.scaler.inverse_transform(center.reshape(1, -1))[0])
            
            cluster_centers = np.array(cluster_centers)
            
            # Create linkage matrix for dendrogram
            linkage_matrix = linkage(scaled_features, method=linkage_method)
            
            # Calculate clustering quality metrics
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            calinski_score = calinski_harabasz_score(scaled_features, cluster_labels)
            davies_bouldin = davies_bouldin_score(scaled_features, cluster_labels)
            
            self.last_algorithm = 'Hierarchical'
            
            st.success(f"""
             **Hierarchical Clustering Complete!**
            - Number of clusters: {n_clusters}
            - Linkage method: {linkage_method}
            - Silhouette score: {silhouette_avg:.3f}
            - Calinski-Harabasz score: {calinski_score:.2f}
            - Davies-Bouldin score: {davies_bouldin:.3f}
            """)
            
            return clustered_df, cluster_centers, linkage_matrix
            
        except Exception as e:
            st.error(f"Error performing hierarchical clustering: {str(e)}")
            raise e
    
    def compare_clustering_algorithms(self, data, feature_columns, n_clusters=4, dbscan_eps=0.5, dbscan_min_samples=5):
        """
        Compare different clustering algorithms on the same dataset.
        
        Args:
            data (pd.DataFrame): Customer data
            feature_columns (list): Columns to use for clustering
            n_clusters (int): Number of clusters for K-Means and Hierarchical
            dbscan_eps (float): DBSCAN epsilon parameter
            dbscan_min_samples (int): DBSCAN min_samples parameter
            
        Returns:
            dict: Results from all algorithms
        """
        try:
            results = {}
            features = data[feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # K-Means++
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=self.random_state, n_init='auto')
            kmeans_labels = kmeans.fit_predict(scaled_features)
            kmeans_silhouette = silhouette_score(scaled_features, kmeans_labels)
            
            results['K-Means++'] = {
                'labels': kmeans_labels,
                'n_clusters': n_clusters,
                'silhouette': kmeans_silhouette,
                'calinski_harabasz': calinski_harabasz_score(scaled_features, kmeans_labels),
                'davies_bouldin': davies_bouldin_score(scaled_features, kmeans_labels)
            }
            
            # DBSCAN
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
            dbscan_labels = dbscan.fit_predict(scaled_features)
            dbscan_n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            if dbscan_n_clusters > 1:
                non_noise_mask = dbscan_labels != -1
                if non_noise_mask.sum() > 0:
                    non_noise_features = scaled_features[non_noise_mask]
                    non_noise_labels = dbscan_labels[non_noise_mask]
                    if len(set(non_noise_labels)) > 1:
                        dbscan_silhouette = silhouette_score(non_noise_features, non_noise_labels)
                        dbscan_calinski = calinski_harabasz_score(non_noise_features, non_noise_labels)
                        dbscan_davies_bouldin = davies_bouldin_score(non_noise_features, non_noise_labels)
                    else:
                        dbscan_silhouette = dbscan_calinski = dbscan_davies_bouldin = 0
                else:
                    dbscan_silhouette = dbscan_calinski = dbscan_davies_bouldin = 0
            else:
                dbscan_silhouette = dbscan_calinski = dbscan_davies_bouldin = 0
            
            results['DBSCAN'] = {
                'labels': dbscan_labels,
                'n_clusters': dbscan_n_clusters,
                'n_noise': (dbscan_labels == -1).sum(),
                'silhouette': dbscan_silhouette,
                'calinski_harabasz': dbscan_calinski,
                'davies_bouldin': dbscan_davies_bouldin
            }
            
            # Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            hierarchical_labels = hierarchical.fit_predict(scaled_features)
            hierarchical_silhouette = silhouette_score(scaled_features, hierarchical_labels)
            
            results['Hierarchical'] = {
                'labels': hierarchical_labels,
                'n_clusters': n_clusters,
                'silhouette': hierarchical_silhouette,
                'calinski_harabasz': calinski_harabasz_score(scaled_features, hierarchical_labels),
                'davies_bouldin': davies_bouldin_score(scaled_features, hierarchical_labels)
            }
            
            st.success("Algorithm comparison completed successfully!")
            return results
            
        except Exception as e:
            st.error(f"Error comparing algorithms: {str(e)}")
            raise e
