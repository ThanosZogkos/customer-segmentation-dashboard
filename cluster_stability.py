import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


class ClusterStabilityAnalyzer:
    
    #Advanced cluster stability analysis and validation metrics.
    
    
    def __init__(self):
        """Initialize the cluster stability analyzer."""
        self.cluster_colors = [
            '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#16213E',
            '#0F3460', '#E94560', '#F7B801', '#00B0FF', '#FF6B35'
        ]
    
    def calculate_comprehensive_metrics(self, data, features, labels, algorithm='K-Means++'):
       
      #  Calculate comprehensive clustering quality metrics.
        
    
        try:
            X = data[features].values
            
            # Filter out noise points for metrics calculation (DBSCAN -1 labels)
            non_noise_mask = labels != -1
            X_filtered = X[non_noise_mask]
            labels_filtered = labels[non_noise_mask]
            
            metrics = {
                'algorithm': algorithm,
                'total_samples': len(labels),
                'noise_samples': np.sum(labels == -1),
                'valid_samples': len(labels_filtered),
                'n_clusters': len(np.unique(labels_filtered)) if len(labels_filtered) > 0 else 0
            }
            
            # Skip metrics if insufficient data
            if len(labels_filtered) < 2 or metrics['n_clusters'] < 2:
                metrics.update({
                    'silhouette_avg': 0,
                    'calinski_harabasz': 0,
                    'davies_bouldin': float('inf'),
                    'inertia': 0,
                    'cluster_separation': 0,
                    'cluster_compactness': 0
                })
                return metrics
            
            # Silhouette Analysis
            silhouette_avg = silhouette_score(X_filtered, labels_filtered)
            metrics['silhouette_avg'] = silhouette_avg
            
            # Calinski-Harabasz Index (Variance Ratio Criterion)
            ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
            metrics['calinski_harabasz'] = ch_score
            
            # Davies-Bouldin Index
            db_score = davies_bouldin_score(X_filtered, labels_filtered)
            metrics['davies_bouldin'] = db_score
            
            # Inertia (only for K-Means-like algorithms)
            if algorithm in ['K-Means++', 'K-Means']:
                try:
                    kmeans_temp = KMeans(n_clusters=metrics['n_clusters'], random_state=42, n_init='auto')
                    kmeans_temp.fit(X_filtered)
                    metrics['inertia'] = kmeans_temp.inertia_
                except:
                    metrics['inertia'] = 0
            else:
                metrics['inertia'] = 0
            
            # Custom metrics
            metrics['cluster_separation'] = self._calculate_cluster_separation(X_filtered, labels_filtered)
            metrics['cluster_compactness'] = self._calculate_cluster_compactness(X_filtered, labels_filtered)
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {
                'algorithm': algorithm,
                'silhouette_avg': 0,
                'calinski_harabasz': 0,
                'davies_bouldin': float('inf'),
                'inertia': 0,
                'cluster_separation': 0,
                'cluster_compactness': 0
            }
    
    def _calculate_cluster_separation(self, X, labels):
        #Calculate average distance between cluster centers.
        
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0
                
            centers = []
            for label in unique_labels:
                cluster_points = X[labels == label]
                center = np.mean(cluster_points, axis=0)
                centers.append(center)
            
            centers = np.array(centers)
            distances = []
            
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    distances.append(dist)
            
            return np.mean(distances) if distances else 0
            
        except:
            return 0
    
    def _calculate_cluster_compactness(self, X, labels):
        #Calculate average within-cluster sum of squares.
        
        try:
            unique_labels = np.unique(labels)
            total_wcss = 0
            total_points = 0
            
            for label in unique_labels:
                cluster_points = X[labels == label]
                if len(cluster_points) > 1:
                    center = np.mean(cluster_points, axis=0)
                    wcss = np.sum((cluster_points - center) ** 2)
                    total_wcss += wcss
                    total_points += len(cluster_points)
            
            return total_wcss / total_points if total_points > 0 else 0
            
        except:
            return 0
    
    def create_silhouette_analysis_plot(self, data, features, labels, algorithm='K-Means++'):
        
       # Create detailed silhouette analysis visualization.
        
        try:
            X = data[features].values
            
            # Filter out noise points
            non_noise_mask = labels != -1
            X_filtered = X[non_noise_mask]
            labels_filtered = labels[non_noise_mask]
            
            if len(labels_filtered) < 2:
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient data for silhouette analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Calculate silhouette scores
            sample_silhouette_values = silhouette_samples(X_filtered, labels_filtered)
            silhouette_avg = silhouette_score(X_filtered, labels_filtered)
            
            fig = go.Figure()
            
            y_lower = 10
            unique_labels = sorted(np.unique(labels_filtered))
            
            for i, cluster_label in enumerate(unique_labels):
                # Get silhouette scores for this cluster
                cluster_silhouette_values = sample_silhouette_values[labels_filtered == cluster_label]
                cluster_silhouette_values.sort()
                
                size_cluster_i = cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                
                color = self.cluster_colors[i % len(self.cluster_colors)]
                
                fig.add_trace(go.Scatter(
                    x=cluster_silhouette_values,
                    y=np.arange(y_lower, y_upper),
                    fill='tonexty' if i > 0 else 'tozeroy',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor=color,
                    name=f'Cluster {cluster_label}',
                    hovertemplate='Cluster %{fullData.name}<br>Silhouette Score: %{x:.3f}<extra></extra>'
                ))
                
                # Label the cluster with its average silhouette score
                cluster_avg = np.mean(cluster_silhouette_values)
                fig.add_annotation(
                    x=cluster_avg,
                    y=y_lower + size_cluster_i / 2,
                    text=f'{cluster_avg:.3f}',
                    showarrow=False,
                    font=dict(size=10, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                )
                
                y_lower = y_upper + 10
            
            # Add vertical line for average silhouette score
            fig.add_vline(
                x=silhouette_avg,
                line=dict(color="red", width=2, dash="dash"),
                annotation_text=f"Average: {silhouette_avg:.3f}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title=f'Silhouette Analysis - {algorithm}',
                xaxis_title='Silhouette Coefficient Values',
                yaxis_title='Cluster Sample Index',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating silhouette analysis: {str(e)}")
            return go.Figure()
    
    def create_cluster_validation_dashboard(self, metrics_list):
        
       # Create comprehensive cluster validation dashboard.
        
        try:
            if not metrics_list:
                return go.Figure()
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Silhouette Score Comparison',
                    'Calinski-Harabasz Index',
                    'Davies-Bouldin Index (Lower is Better)',
                    'Cluster Quality Summary'
                ],
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "table"}]]
            )
            
            algorithms = [m['algorithm'] for m in metrics_list]
            colors = self.cluster_colors[:len(algorithms)]
            
            # Silhouette scores
            silhouette_scores = [m['silhouette_avg'] for m in metrics_list]
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=silhouette_scores,
                    marker_color=colors,
                    name='Silhouette',
                    text=[f'{score:.3f}' for score in silhouette_scores],
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Calinski-Harabasz scores
            ch_scores = [m['calinski_harabasz'] for m in metrics_list]
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=ch_scores,
                    marker_color=colors,
                    name='CH Index',
                    text=[f'{score:.1f}' for score in ch_scores],
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            # Davies-Bouldin scores
            db_scores = [m['davies_bouldin'] for m in metrics_list]
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=db_scores,
                    marker_color=colors,
                    name='DB Index',
                    text=[f'{score:.3f}' for score in db_scores],
                    textposition='auto'
                ),
                row=2, col=1
            )
            
            # Summary table
            table_data = []
            for m in metrics_list:
                table_data.append([
                    m['algorithm'],
                    f"{m['n_clusters']}",
                    f"{m['silhouette_avg']:.3f}",
                    f"{m['calinski_harabasz']:.1f}",
                    f"{m['davies_bouldin']:.3f}",
                    f"{m.get('noise_samples', 0)}"
                ])
            
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=['Algorithm', 'Clusters', 'Silhouette', 'CH Index', 'DB Index', 'Noise'],
                        fill_color='#2E86AB',
                        font=dict(color='white', size=12),
                        align='center'
                    ),
                    cells=dict(
                        values=list(zip(*table_data)),
                        fill_color='#f8f9fa',
                        align='center',
                        font=dict(size=11)
                    )
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title='Comprehensive Cluster Validation Dashboard',
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=700,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating validation dashboard: {str(e)}")
            return go.Figure()
    
    def recommend_optimal_algorithm(self, metrics_list):
        
       # Recommend the optimal clustering algorithm based on metrics.
        
        if not metrics_list:
            return {
                "algorithm": "None", 
                "reasoning": "No metrics available",
                "score": 0,
                "all_scores": {}
            }
        
        # Scoring system (normalized between 0-1)
        scores = {}
        
        for metrics in metrics_list:
            algo = metrics['algorithm']
            score = 0
            criteria = []
            
            # Silhouette score (higher is better, range -1 to 1, normalize to 0-1)
            silhouette_norm = (metrics['silhouette_avg'] + 1) / 2
            score += silhouette_norm * 0.4
            criteria.append(f"Silhouette: {silhouette_norm:.3f}")
            
            # Calinski-Harabasz (higher is better, normalize by max)
            max_ch = max([m['calinski_harabasz'] for m in metrics_list])
            if max_ch > 0:
                ch_norm = metrics['calinski_harabasz'] / max_ch
                score += ch_norm * 0.3
                criteria.append(f"CH: {ch_norm:.3f}")
            
            # Davies-Bouldin (lower is better, invert and normalize)
            min_db = min([m['davies_bouldin'] for m in metrics_list if m['davies_bouldin'] != float('inf')])
            if min_db > 0 and metrics['davies_bouldin'] != float('inf'):
                db_norm = min_db / metrics['davies_bouldin']
                score += db_norm * 0.2
                criteria.append(f"DB: {db_norm:.3f}")
            
            # Penalty for noise points (DBSCAN)
            if metrics.get('noise_samples', 0) > 0:
                noise_penalty = metrics['noise_samples'] / metrics['total_samples']
                score -= noise_penalty * 0.1
                criteria.append(f"Noise penalty: -{noise_penalty:.3f}")
            
            scores[algo] = {
                'score': score,
                'criteria': criteria,
                'metrics': metrics
            }
        
        # Find best algorithm
        best_algo = max(scores.keys(), key=lambda x: scores[x]['score'])
        best_score = scores[best_algo]
        
        reasoning = f"""
        **Recommended Algorithm: {best_algo}**
        
        **Overall Score:** {best_score['score']:.3f}
        
        **Evaluation Criteria:**
        - {' | '.join(best_score['criteria'])}
        
        **Key Metrics:**
        - Silhouette Score: {best_score['metrics']['silhouette_avg']:.3f}
        - Number of Clusters: {best_score['metrics']['n_clusters']}
        - Calinski-Harabasz Index: {best_score['metrics']['calinski_harabasz']:.1f}
        - Davies-Bouldin Index: {best_score['metrics']['davies_bouldin']:.3f}
        """
        
        if best_score['metrics'].get('noise_samples', 0) > 0:
            reasoning += f"\n- Noise Points: {best_score['metrics']['noise_samples']}"
        
        return {
            'algorithm': best_algo,
            'score': best_score['score'],
            'reasoning': reasoning,
            'all_scores': scores
        }
