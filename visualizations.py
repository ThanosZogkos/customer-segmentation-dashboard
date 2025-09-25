import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class VisualizationGenerator:
    """
    Generates interactive visualizations for customer segmentation analysis.
    """
    
    def __init__(self):
        # Color palette for consistent styling
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#06D6A0',
            'background': '#F8F9FA',
            'text': '#212529'
        }
        
        # Cluster colors for visualization
        self.cluster_colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#06D6A0', 
            '#F72585', '#7209B7', '#560BAD', '#480CA8'
        ]
    
    def create_elbow_plot(self, k_values, inertias, optimal_k):
        """
        Create elbow plot for optimal cluster determination.
        
        Args:
            k_values (range): Range of k values tested
            inertias (list): Corresponding inertia values
            optimal_k (int): Optimal number of clusters
            
        Returns:
            plotly.graph_objects.Figure: Elbow plot
        """
        try:
            fig = go.Figure()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=list(k_values),
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=8, color=self.colors['primary'])
            ))
            
            # Highlight optimal k
            if optimal_k in k_values:
                optimal_index = list(k_values).index(optimal_k)
                fig.add_trace(go.Scatter(
                    x=[optimal_k],
                    y=[inertias[optimal_index]],
                    mode='markers',
                    name=f'Optimal K ({optimal_k})',
                    marker=dict(size=15, color=self.colors['accent'], symbol='star')
                ))
            
            fig.update_layout(
                title='Elbow Method for Optimal Number of Clusters',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Inertia',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                showlegend=True,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating elbow plot: {str(e)}")
            return go.Figure()
    
    def create_silhouette_plot(self, k_values, silhouette_scores):
        """
        Create silhouette score plot for cluster validation.
        
        Args:
            k_values (range): Range of k values tested
            silhouette_scores (list): Corresponding silhouette scores
            
        Returns:
            plotly.graph_objects.Figure: Silhouette plot
        """
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(k_values),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color=self.colors['secondary'], width=3),
                marker=dict(size=8, color=self.colors['secondary'])
            ))
            
            # Find and highlight best silhouette score
            if silhouette_scores:
                best_score_idx = np.argmax(silhouette_scores)
                best_k = list(k_values)[best_score_idx]
                
                fig.add_trace(go.Scatter(
                    x=[best_k],
                    y=[silhouette_scores[best_score_idx]],
                    mode='markers',
                    name=f'Best Score (K={best_k})',
                    marker=dict(size=15, color=self.colors['success'], symbol='star')
                ))
            
            fig.update_layout(
                title='Silhouette Score Analysis',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Silhouette Score',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                showlegend=True,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating silhouette plot: {str(e)}")
            return go.Figure()
    
    def create_cluster_scatter(self, data, x_col, y_col, cluster_col, cluster_centers=None):
        """
        Create 2D scatter plot showing customer clusters with centroids.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis
            cluster_col (str): Column name for cluster assignments
            cluster_centers (np.ndarray): Cluster centers coordinates (optional)
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot with centroids
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Get feature columns mapping
            feature_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            
            # Find indices for x and y columns
            x_idx = feature_cols.index(x_col) if x_col in feature_cols else None
            y_idx = feature_cols.index(y_col) if y_col in feature_cols else None
            
            # Get unique clusters (excluding noise points)
            unique_clusters = sorted([c for c in data[cluster_col].unique() if c != -1])
            
            # Add cluster data points
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data[cluster_col] == cluster]
                cluster_size = len(cluster_data)
                
                fig.add_trace(go.Scatter(
                    x=cluster_data[x_col],
                    y=cluster_data[y_col],
                    mode='markers',
                    name=f'Cluster {cluster} ({cluster_size} customers)',
                    marker=dict(
                        size=8,
                        color=self.cluster_colors[i % len(self.cluster_colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        '<b>Customer %{customdata[0]}</b><br>' +
                        f'{x_col}: %{{x}}<br>' +
                        f'{y_col}: %{{y}}<br>' +
                        'Gender: %{customdata[1]}<br>' +
                        'Age: %{customdata[2]}<br>' +
                        'Income: $%{customdata[3]}k<br>' +
                        'Spending: %{customdata[4]}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=cluster_data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
                ))
            
            # Add noise points if they exist (DBSCAN)
            if -1 in data[cluster_col].unique():
                noise_data = data[data[cluster_col] == -1]
                if len(noise_data) > 0:
                    fig.add_trace(go.Scatter(
                        x=noise_data[x_col],
                        y=noise_data[y_col],
                        mode='markers',
                        name=f'Noise Points ({len(noise_data)} customers)',
                        marker=dict(
                            size=6,
                            color='gray',
                            opacity=0.4,
                            symbol='x',
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=(
                            '<b>Customer %{customdata[0]}</b><br>' +
                            f'{x_col}: %{{x}}<br>' +
                            f'{y_col}: %{{y}}<br>' +
                            'Gender: %{customdata[1]}<br>' +
                            'Status: Noise Point<br>' +
                            '<extra></extra>'
                        ),
                        customdata=noise_data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
                    ))
            
            # Add cluster centroids if provided
            if cluster_centers is not None and x_idx is not None and y_idx is not None:
                if cluster_centers.ndim == 2 and cluster_centers.shape[1] >= max(x_idx + 1, y_idx + 1):
                    centroid_x = cluster_centers[:len(unique_clusters), x_idx]
                    centroid_y = cluster_centers[:len(unique_clusters), y_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=centroid_x,
                        y=centroid_y,
                        mode='markers',
                        name='Cluster Centroids',
                        marker=dict(
                            size=15,
                            color='black',
                            symbol='diamond',
                            line=dict(width=3, color='white')
                        ),
                        hovertemplate=(
                            '<b>Cluster Centroid</b><br>' +
                            f'{x_col}: %{{x:.1f}}<br>' +
                            f'{y_col}: %{{y:.1f}}<br>' +
                            '<extra></extra>'
                        )
                    ))
            
            fig.update_layout(
                title=f'Customer Segments: {x_col} vs {y_col}',
                xaxis_title=x_col,
                yaxis_title=y_col,
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
            return go.Figure()
    
    def create_3d_cluster_plot(self, data, cluster_centers=None, show_centers=True):
        """
        Create enhanced 3D scatter plot for comprehensive cluster visualization.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            cluster_centers (np.ndarray): Cluster centers coordinates
            show_centers (bool): Whether to show cluster centers
            
        Returns:
            plotly.graph_objects.Figure: Enhanced 3D scatter plot
        """
        try:
            unique_clusters = sorted(data['Cluster'].unique())
            
            fig = go.Figure()
            
            # Add cluster data points
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                cluster_size = len(cluster_data)
                
                # Handle noise points in DBSCAN (cluster -1)
                if cluster == -1:
                    cluster_name = 'Noise Points'
                    marker_color = 'gray'
                    marker_symbol = 'x'
                    opacity = 0.4
                else:
                    cluster_name = f'Cluster {cluster} ({cluster_size} customers)'
                    marker_color = self.cluster_colors[i % len(self.cluster_colors)]
                    marker_symbol = 'circle'
                    opacity = 0.7
                
                fig.add_trace(go.Scatter3d(
                    x=cluster_data['Age'],
                    y=cluster_data['Annual Income (k$)'],
                    z=cluster_data['Spending Score (1-100)'],
                    mode='markers',
                    name=cluster_name,
                    marker=dict(
                        size=6 if cluster != -1 else 4,
                        color=marker_color,
                        opacity=opacity,
                        symbol=marker_symbol,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=(
                        '<b>Customer %{customdata[0]}</b><br>' +
                        'Age: %{x}<br>' +
                        'Income: $%{y}k<br>' +
                        'Spending Score: %{z}<br>' +
                        'Gender: %{customdata[1]}<br>' +
                        f'Cluster: {cluster_name}<br>' +
                        '<extra></extra>'
                    ),
                    customdata=cluster_data[['CustomerID', 'Gender']].values
                ))
            
            # Add cluster centers if provided and valid
            if show_centers and cluster_centers is not None and len(cluster_centers) > 0:
                # Ensure cluster_centers has at least 3 columns
                if cluster_centers.ndim == 2 and cluster_centers.shape[1] >= 3:
                    center_ages = cluster_centers[:, 0]
                    center_incomes = cluster_centers[:, 1] 
                    center_spending = cluster_centers[:, 2]
                    
                    fig.add_trace(go.Scatter3d(
                        x=center_ages,
                        y=center_incomes,
                        z=center_spending,
                        mode='markers',
                        name='Cluster Centers',
                        marker=dict(
                            size=12,
                            color='black',
                            symbol='diamond',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=(
                            '<b>Cluster Center</b><br>' +
                            'Avg Age: %{x:.1f}<br>' +
                            'Avg Income: $%{y:.1f}k<br>' +
                            'Avg Spending: %{z:.1f}<br>' +
                            '<extra></extra>'
                        )
                    ))
            
            fig.update_layout(
                title='Enhanced 3D Customer Segmentation Visualization',
                scene=dict(
                    xaxis=dict(
                        title='Age (years)',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title='Annual Income (k$)',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    zaxis=dict(
                        title='Spending Score (1-100)',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.5),
                        center=dict(x=0, y=0, z=0)
                    ),
                    aspectmode='cube'
                ),
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=0
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating 3D plot: {str(e)}")
            return go.Figure()
    
    def create_multidimensional_analysis_plot(self, data, features):
        """
        Create multi-dimensional analysis with parallel coordinates and correlation matrix.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            features (list): Features to analyze
            
        Returns:
            tuple: (parallel_coordinates_fig, correlation_heatmap_fig)
        """
        try:
            # Parallel coordinates plot
            feature_data = data[features + ['Cluster']].copy()
            unique_clusters = sorted(data['Cluster'].unique())
            
            parallel_fig = go.Figure()
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = feature_data[feature_data['Cluster'] == cluster]
                if cluster == -1:
                    cluster_name = 'Noise Points'
                    color = 'gray'
                else:
                    cluster_name = f'Cluster {cluster}'
                    color = self.cluster_colors[i % len(self.cluster_colors)]
                
                # Normalize features for parallel coordinates
                normalized_data = cluster_data[features].copy()
                for feature in features:
                    min_val = data[feature].min()
                    max_val = data[feature].max()
                    if max_val == min_val:
                        # Handle constant features
                        normalized_data[feature] = 0.5
                    else:
                        normalized_data[feature] = (normalized_data[feature] - min_val) / (max_val - min_val)
                
                dimensions = []
                for j, feature in enumerate(features):
                    dimensions.append(dict(
                        range=[0, 1],
                        label=feature,
                        values=normalized_data[feature]
                    ))
                
                parallel_fig.add_trace(go.Parcoords(
                    line=dict(
                        color=color,
                        colorscale=[[0, color], [1, color]]
                    ),
                    dimensions=dimensions,
                    name=cluster_name
                ))
            
            parallel_fig.update_layout(
                title='Multi-Dimensional Feature Analysis (Parallel Coordinates)',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=500
            )
            
            # Correlation heatmap
            correlation_matrix = data[features].corr()
            
            correlation_fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=features,
                y=features,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate='%{text}',
                textfont={'size': 12},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            correlation_fig.update_layout(
                title='Feature Correlation Matrix',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=400,
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')
            )
            
            return parallel_fig, correlation_fig
            
        except Exception as e:
            st.error(f"Error creating multi-dimensional analysis: {str(e)}")
            return go.Figure(), go.Figure()
    
    def create_cluster_volume_analysis(self, data, features):
        """
        Create 3D volume analysis showing cluster density and boundaries.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            features (list): Three features for 3D analysis
            
        Returns:
            plotly.graph_objects.Figure: 3D volume analysis plot
        """
        try:
            if len(features) != 3:
                st.warning("Volume analysis requires exactly 3 features. Using first 3.")
                features = features[:3]
            
            unique_clusters = sorted([c for c in data['Cluster'].unique() if c != -1])  # Exclude noise
            
            fig = go.Figure()
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                
                # Create convex hull approximation using cluster boundaries
                x_data = cluster_data[features[0]]
                y_data = cluster_data[features[1]]
                z_data = cluster_data[features[2]]
                
                # Add scatter points
                fig.add_trace(go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=6,
                        color=self.cluster_colors[i % len(self.cluster_colors)],
                        opacity=0.6
                    )
                ))
                
                # Add cluster boundary ellipsoid (simplified)
                if len(cluster_data) > 4:  # Need minimum points for ellipsoid
                    # Calculate cluster statistics
                    center_x, center_y, center_z = x_data.mean(), y_data.mean(), z_data.mean()
                    std_x, std_y, std_z = x_data.std(), y_data.std(), z_data.std()
                    
                    # Create ellipsoid surface points
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 10)
                    
                    x_ellipsoid = center_x + 2 * std_x * np.outer(np.cos(u), np.sin(v)).flatten()
                    y_ellipsoid = center_y + 2 * std_y * np.outer(np.sin(u), np.sin(v)).flatten()
                    z_ellipsoid = center_z + 2 * std_z * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
                    
                    fig.add_trace(go.Mesh3d(
                        x=x_ellipsoid,
                        y=y_ellipsoid,
                        z=z_ellipsoid,
                        alphahull=5,
                        opacity=0.1,
                        color=self.cluster_colors[i % len(self.cluster_colors)],
                        showscale=False,
                        name=f'Cluster {cluster} Boundary',
                        showlegend=False
                    ))
            
            fig.update_layout(
                title='3D Cluster Volume and Boundary Analysis',
                scene=dict(
                    xaxis_title=features[0],
                    yaxis_title=features[1],
                    zaxis_title=features[2],
                    camera=dict(eye=dict(x=2, y=2, z=2))
                ),
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating volume analysis: {str(e)}")
            return go.Figure()
    
    def create_cluster_distribution(self, data):
        """
        Create pie chart showing cluster size distribution.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            
        Returns:
            plotly.graph_objects.Figure: Pie chart
        """
        try:
            cluster_counts = data['Cluster'].value_counts().sort_index()
            
            fig = go.Figure(data=[go.Pie(
                labels=[f'Cluster {i}' for i in cluster_counts.index],
                values=cluster_counts.values,
                marker_colors=[self.cluster_colors[i % len(self.cluster_colors)] 
                              for i in range(len(cluster_counts))],
                textinfo='label+percent+value',
                textposition='auto',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title='Customer Distribution Across Clusters',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            return go.Figure()
    
    def create_gender_distribution_by_cluster(self, data):
        """
        Create stacked bar chart showing gender distribution by cluster.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            
        Returns:
            plotly.graph_objects.Figure: Stacked bar chart
        """
        try:
            # Calculate gender distribution by cluster
            gender_cluster = data.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            
            for gender in gender_cluster.columns:
                fig.add_trace(go.Bar(
                    name=gender,
                    x=[f'Cluster {i}' for i in gender_cluster.index],
                    y=gender_cluster[gender],
                    marker_color=self.colors['primary'] if gender == 'Male' else self.colors['secondary']
                ))
            
            fig.update_layout(
                title='Gender Distribution by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Number of Customers',
                barmode='stack',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating gender distribution plot: {str(e)}")
            return go.Figure()
    
    def create_feature_comparison_by_cluster(self, data, features):
        """
        Create radar chart comparing features across clusters.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            features (list): Features to compare
            
        Returns:
            plotly.graph_objects.Figure: Radar chart
        """
        try:
            fig = go.Figure()
            
            unique_clusters = sorted(data['Cluster'].unique())
            
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                
                # Calculate normalized means for radar chart
                values = []
                for feature in features:
                    mean_val = cluster_data[feature].mean()
                    # Normalize to 0-100 scale based on overall data range
                    min_val = data[feature].min()
                    max_val = data[feature].max()
                    normalized = ((mean_val - min_val) / (max_val - min_val)) * 100
                    values.append(normalized)
                
                # Close the radar chart
                values.append(values[0])
                feature_labels = features + [features[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=feature_labels,
                    fill='toself',
                    name=f'Cluster {cluster}',
                    line_color=self.cluster_colors[i % len(self.cluster_colors)]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title='Feature Comparison Across Clusters',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=500
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating radar chart: {str(e)}")
            return go.Figure()
    
    def create_cluster_heatmap(self, data, features):
        """
        Create heatmap showing average feature values by cluster.
        
        Args:
            data (pd.DataFrame): Clustered customer data
            features (list): Features to display
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        try:
            # Calculate average values by cluster
            cluster_means = data.groupby('Cluster')[features].mean()
            
            fig = go.Figure(data=go.Heatmap(
                z=cluster_means.values,
                x=features,
                y=[f'Cluster {i}' for i in cluster_means.index],
                colorscale='Blues',
                showscale=True,
                text=cluster_means.round(1).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate='Cluster: %{y}<br>Feature: %{x}<br>Average: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Average Feature Values by Cluster',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
            return go.Figure()
    
    def create_dendrogram_plot(self, linkage_matrix, feature_names):
        """
        Create dendrogram visualization for hierarchical clustering.
        
        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering
            feature_names (list): Names of features used for clustering
            
        Returns:
            str: Base64 encoded dendrogram image
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create dendrogram
            dendrogram_data = dendrogram(
                linkage_matrix,
                ax=ax,
                leaf_rotation=90,
                leaf_font_size=10,
                color_threshold=0,
                above_threshold_color='gray'
            )
            
            ax.set_title('Hierarchical Clustering Dendrogram', fontsize=16, fontweight='bold')
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('Distance', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Convert to base64 string for display
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            image_png = buffer.getvalue()
            buffer.close()
            plt.close(fig)
            
            graphic = base64.b64encode(image_png)
            graphic = graphic.decode('utf-8')
            
            return graphic
            
        except Exception as e:
            st.error(f"Error creating dendrogram: {str(e)}")
            return ""
    
    def create_algorithm_comparison_plot(self, comparison_results):
        """
        Create comparison visualization for different clustering algorithms.
        
        Args:
            comparison_results (dict): Results from algorithm comparison
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        try:
            algorithms = list(comparison_results.keys())
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Number of Clusters', 'Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Number of clusters comparison
            n_clusters_data = []
            for alg in algorithms:
                n_clusters_data.append(comparison_results[alg]['n_clusters'])
            
            fig.add_trace(go.Bar(
                x=algorithms,
                y=n_clusters_data,
                name='Clusters',
                marker_color=self.colors['primary'],
                showlegend=False
            ), row=1, col=1)
            
            # Silhouette score comparison
            silhouette_data = []
            for alg in algorithms:
                silhouette_data.append(comparison_results[alg]['silhouette'])
            
            fig.add_trace(go.Bar(
                x=algorithms,
                y=silhouette_data,
                name='Silhouette',
                marker_color=self.colors['secondary'],
                showlegend=False
            ), row=1, col=2)
            
            # Calinski-Harabasz score comparison
            calinski_data = []
            for alg in algorithms:
                calinski_data.append(comparison_results[alg]['calinski_harabasz'])
            
            fig.add_trace(go.Bar(
                x=algorithms,
                y=calinski_data,
                name='Calinski-Harabasz',
                marker_color=self.colors['accent'],
                showlegend=False
            ), row=2, col=1)
            
            # Davies-Bouldin score comparison (lower is better)
            davies_bouldin_data = []
            for alg in algorithms:
                davies_bouldin_data.append(comparison_results[alg]['davies_bouldin'])
            
            fig.add_trace(go.Bar(
                x=algorithms,
                y=davies_bouldin_data,
                name='Davies-Bouldin',
                marker_color=self.colors['success'],
                showlegend=False
            ), row=2, col=2)
            
            fig.update_layout(
                title='Clustering Algorithm Comparison',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=600,
                showlegend=False
            )
            
            # Update y-axes labels
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Score", row=1, col=2)
            fig.update_yaxes(title_text="Score", row=2, col=1)
            fig.update_yaxes(title_text="Score (Lower=Better)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating algorithm comparison plot: {str(e)}")
            return go.Figure()
    
    def create_cluster_stability_plot(self, stability_scores, k_range):
        """
        Create cluster stability analysis visualization.
        
        Args:
            stability_scores (list): Stability scores for different k values
            k_range (range): Range of k values tested
            
        Returns:
            plotly.graph_objects.Figure: Stability plot
        """
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(k_range),
                y=stability_scores,
                mode='lines+markers',
                name='Stability Score',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=10, color=self.colors['primary'])
            ))
            
            # Highlight most stable k
            if stability_scores:
                best_k_idx = np.argmax(stability_scores)
                best_k = list(k_range)[best_k_idx]
                
                fig.add_trace(go.Scatter(
                    x=[best_k],
                    y=[stability_scores[best_k_idx]],
                    mode='markers',
                    name=f'Most Stable (K={best_k})',
                    marker=dict(size=15, color=self.colors['accent'], symbol='star')
                ))
            
            fig.update_layout(
                title='Cluster Stability Analysis',
                xaxis_title='Number of Clusters (K)',
                yaxis_title='Stability Score',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                showlegend=True,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating stability plot: {str(e)}")
            return go.Figure()
