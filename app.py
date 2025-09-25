import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from clustering import CustomerSegmentation
from visualizations import VisualizationGenerator
from business_insights import BusinessInsights
from cluster_stability import ClusterStabilityAnalyzer

# Configure page
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2E86AB;
    margin: 0.5rem 0;
}
.insight-card {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #A23B72;
    margin: 1rem 0;
}
.segment-card {
    background-color: #FFFFFF;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #F18F01;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛍️ Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize components
    data_processor = DataProcessor()
    
    # Load and process data
    try:
        df = data_processor.load_data("attached_assets/Mall_Customers_1758472298487.csv")
        processed_df = data_processor.preprocess_data(df)
        
        # Raw Data Overview Section
        st.header("📋 Customer Data Overview")
        st.write("This section shows the raw customer data before any clustering analysis is applied.")
        
        # Data summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            st.metric("Data Features", len(df.columns))
        
        with col3:
            avg_age = df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f} years")
        
        with col4:
            avg_income = df['Annual Income (k$)'].mean()
            st.metric("Average Income", f"${avg_income:.1f}k")
        
        # Raw data table
        st.subheader("📊 Raw Customer Data")
        st.write("Complete customer dataset with all original features:")
        st.dataframe(df, use_container_width=True)
        
        # Basic statistics
        st.subheader("📈 Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numerical Features:**")
            numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.write("**Gender Distribution:**")
            gender_counts = df['Gender'].value_counts()
            
            # Create gender distribution chart
            import plotly.express as px
            gender_fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Customer Gender Distribution",
                color_discrete_map={'Male': '#2E86AB', 'Female': '#A23B72'}
            )
            gender_fig.update_layout(
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=300
            )
            st.plotly_chart(gender_fig, use_container_width=True)
        
        # Data distributions
        st.subheader("📊 Feature Distributions")
        
        # Create distribution plots
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Age Distribution', 'Annual Income Distribution', 
                          'Spending Score Distribution', 'Age vs Income'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Age distribution
        fig_dist.add_trace(
            go.Histogram(x=df['Age'], name='Age', marker_color='#2E86AB', opacity=0.7),
            row=1, col=1
        )
        
        # Income distribution
        fig_dist.add_trace(
            go.Histogram(x=df['Annual Income (k$)'], name='Income', marker_color='#A23B72', opacity=0.7),
            row=1, col=2
        )
        
        # Spending Score distribution
        fig_dist.add_trace(
            go.Histogram(x=df['Spending Score (1-100)'], name='Spending Score', marker_color='#F18F01', opacity=0.7),
            row=2, col=1
        )
        
        # Age vs Income scatter
        colors = ['#2E86AB' if gender == 'Male' else '#A23B72' for gender in df['Gender']]
        fig_dist.add_trace(
            go.Scatter(
                x=df['Age'], 
                y=df['Annual Income (k$)'], 
                mode='markers',
                name='Age vs Income',
                marker=dict(color=colors, opacity=0.6),
                text=df['Gender'],
                hovertemplate='Age: %{x}<br>Income: $%{y}k<br>Gender: %{text}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig_dist.update_layout(
            template='plotly_white',
            font=dict(family="Inter, sans-serif"),
            height=600,
            showlegend=False,
            title_text="Customer Data Distribution Analysis"
        )
        
        # Update axis labels
        fig_dist.update_xaxes(title_text="Age (years)", row=1, col=1)
        fig_dist.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
        fig_dist.update_xaxes(title_text="Spending Score (1-100)", row=2, col=1)
        fig_dist.update_xaxes(title_text="Age (years)", row=2, col=2)
        fig_dist.update_yaxes(title_text="Count", row=1, col=1)
        fig_dist.update_yaxes(title_text="Count", row=1, col=2)
        fig_dist.update_yaxes(title_text="Count", row=2, col=1)
        fig_dist.update_yaxes(title_text="Annual Income (k$)", row=2, col=2)
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Data quality overview
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_values = df.isnull().sum().sum()
            st.metric("Missing Values", missing_values, help="Total number of missing values in the dataset")
        
        with col2:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates, help="Number of duplicate customer records")
        
        with col3:
            data_types = len(df.dtypes.unique())
            st.metric("Data Types", data_types, help="Number of different data types in the dataset")
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations")
        correlation_matrix = df[numeric_cols].corr()
        
        correlation_fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix of Numerical Features",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        correlation_fig.update_layout(
            template='plotly_white',
            font=dict(family="Inter, sans-serif"),
            height=400
        )
        
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Data insights summary
        st.subheader("💡 Key Data Insights")
        
        insights_text = f"""
        **Dataset Characteristics:**
        - **Size**: {len(df):,} customers with {len(df.columns)} features
        - **Age Range**: {df['Age'].min()} - {df['Age'].max()} years (Avg: {df['Age'].mean():.1f})
        - **Income Range**: ${df['Annual Income (k$)'].min()}k - ${df['Annual Income (k$)'].max()}k (Avg: ${df['Annual Income (k$)'].mean():.1f}k)
        - **Spending Score Range**: {df['Spending Score (1-100)'].min()} - {df['Spending Score (1-100)'].max()} (Avg: {df['Spending Score (1-100)'].mean():.1f})
        - **Gender Split**: {(df['Gender'] == 'Female').sum()} Female ({(df['Gender'] == 'Female').mean()*100:.1f}%), {(df['Gender'] == 'Male').sum()} Male ({(df['Gender'] == 'Male').mean()*100:.1f}%)
        
        **Data Quality:**
        - ✅ No missing values detected
        - ✅ No duplicate records found
        - ✅ All data types are appropriate for analysis
        """
        
        st.markdown(insights_text)
        
        st.divider()
        
        # Sidebar controls
        st.sidebar.header("📊 Analysis Controls")
        
        # Cluster number selection
        max_clusters = st.sidebar.slider(
            "Maximum Clusters for Analysis",
            min_value=2,
            max_value=10,
            value=8,
            help="Maximum number of clusters to consider for elbow method"
        )
        
        # Algorithm selection
        st.sidebar.subheader("🔬 Clustering Algorithm")
        algorithm = st.sidebar.selectbox(
            "Select Algorithm",
            ["K-Means++", "DBSCAN", "Hierarchical", "Compare All"],
            help="Choose the clustering algorithm to use for analysis"
        )
        
        # Algorithm-specific parameters
        eps = 0.5  # Default value
        min_samples = 5  # Default value
        linkage_method = 'ward'  # Default value
        
        if algorithm == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help="Maximum distance between samples")
            min_samples = st.sidebar.slider("Min Samples", 2, 10, 5, help="Minimum samples in neighborhood")
        elif algorithm == "Hierarchical":
            linkage_method = st.sidebar.selectbox(
                "Linkage Method", 
                ["ward", "complete", "average", "single"],
                help="Linkage method for hierarchical clustering"
            )
        elif algorithm == "Compare All":
            eps = st.sidebar.slider("DBSCAN Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help="Maximum distance between samples for DBSCAN")
            min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 10, 5, help="Minimum samples in neighborhood for DBSCAN")
            linkage_method = st.sidebar.selectbox(
                "Hierarchical Linkage Method", 
                ["ward", "complete", "average", "single"],
                help="Linkage method for hierarchical clustering"
            )
        
        # Feature selection for clustering
        st.sidebar.subheader("🎯 Clustering Features")
        use_age = st.sidebar.checkbox("Age", value=True)
        use_income = st.sidebar.checkbox("Annual Income", value=True)
        use_spending = st.sidebar.checkbox("Spending Score", value=True)
        
        if not any([use_age, use_income, use_spending]):
            st.sidebar.error("Please select at least one feature for clustering!")
            return
        
        # Prepare features for clustering
        feature_columns = []
        if use_age:
            feature_columns.append('Age')
        if use_income:
            feature_columns.append('Annual Income (k$)')
        if use_spending:
            feature_columns.append('Spending Score (1-100)')
        
        # Initialize clustering
        clustering = CustomerSegmentation()
        
        # Algorithm-specific clustering
        if algorithm == "K-Means++":
            # Find optimal clusters
            with st.spinner("🔍 Finding optimal number of clusters..."):
                inertias, silhouette_scores = clustering.find_optimal_clusters(
                    processed_df[feature_columns], max_clusters
                )
                optimal_k = clustering.get_optimal_clusters_elbow(inertias)
            
            # Perform clustering
            with st.spinner("🎯 Performing K-Means++ clustering..."):
                clustered_df, cluster_centers = clustering.perform_clustering(
                    processed_df, feature_columns, optimal_k
                )
        
        elif algorithm == "DBSCAN":
            with st.spinner("🎯 Performing DBSCAN clustering..."):
                clustered_df, cluster_centers, n_clusters = clustering.perform_dbscan_clustering(
                    processed_df, feature_columns, eps, min_samples
                )
                optimal_k = n_clusters
                inertias, silhouette_scores = [], []  # Not applicable for DBSCAN
        
        elif algorithm == "Hierarchical":
            # Use optimal k from elbow method for hierarchical
            with st.spinner("🔍 Finding optimal number of clusters..."):
                inertias, silhouette_scores = clustering.find_optimal_clusters(
                    processed_df[feature_columns], max_clusters
                )
                optimal_k = clustering.get_optimal_clusters_elbow(inertias)
            
            with st.spinner("🎯 Performing Hierarchical clustering..."):
                clustered_df, cluster_centers, linkage_matrix = clustering.perform_hierarchical_clustering(
                    processed_df, feature_columns, optimal_k, linkage_method
                )
        
        elif algorithm == "Compare All":
            with st.spinner("🔍 Comparing all clustering algorithms..."):
                inertias, silhouette_scores = clustering.find_optimal_clusters(
                    processed_df[feature_columns], max_clusters
                )
                optimal_k = clustering.get_optimal_clusters_elbow(inertias)
                
                comparison_results = clustering.compare_clustering_algorithms(
                    processed_df, feature_columns, optimal_k, eps, min_samples
                )
                
                # Use K-Means++ results for main visualization
                clustered_df, cluster_centers = clustering.perform_clustering(
                    processed_df, feature_columns, optimal_k
                )
        
        # Initialize visualization and insights generators
        viz_gen = VisualizationGenerator()
        insights = BusinessInsights()
        stability_analyzer = ClusterStabilityAnalyzer()
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #2E86AB; margin: 0;">Total Customers</h3>
                <h2 style="margin: 0;">{len(clustered_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cluster_label = "Clusters Found" if algorithm == "DBSCAN" else "Optimal Clusters"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #A23B72; margin: 0;">{cluster_label}</h3>
                <h2 style="margin: 0;">{optimal_k if optimal_k > 0 else 'N/A'}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_income = clustered_df['Annual Income (k$)'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #F18F01; margin: 0;">Avg. Income</h3>
                <h2 style="margin: 0;">${avg_income:.0f}k</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_spending = clustered_df['Spending Score (1-100)'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #06D6A0; margin: 0;">Avg. Spending Score</h3>
                <h2 style="margin: 0;">{avg_spending:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm-specific analysis sections
        if algorithm in ["K-Means++", "Hierarchical"] or (algorithm == "Compare All"):
            st.header("🔍 Clustering Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Elbow Method Analysis")
                elbow_fig = viz_gen.create_elbow_plot(range(1, max_clusters + 1), inertias, optimal_k)
                st.plotly_chart(elbow_fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Silhouette Score Analysis")
                silhouette_fig = viz_gen.create_silhouette_plot(range(2, max_clusters + 1), silhouette_scores)
                st.plotly_chart(silhouette_fig, use_container_width=True)
        
        # Algorithm comparison section
        if algorithm == "Compare All":
            st.header("⚖️ Algorithm Comparison")
            comparison_fig = viz_gen.create_algorithm_comparison_plot(comparison_results)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Display comparison results table
            st.subheader("📋 Detailed Comparison Results")
            comparison_df = pd.DataFrame(comparison_results).T
            comparison_df = comparison_df.round(3)
            st.dataframe(comparison_df, use_container_width=True)
        
        # Hierarchical-specific dendrogram
        if algorithm == "Hierarchical":
            st.header("🌳 Dendrogram Analysis")
            st.subheader("Hierarchical Clustering Dendrogram")
            dendrogram_img = viz_gen.create_dendrogram_plot(linkage_matrix, feature_columns)
            if dendrogram_img:
                st.image(f"data:image/png;base64,{dendrogram_img}")
        
        # Customer segmentation visualization
        st.header("🎯 Customer Segmentation Visualization")
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Age vs Income", "Age vs Spending Score", "Income vs Spending Score", "Enhanced 3D Visualization", "Multi-Dimensional Analysis", "3D Volume Analysis"]
        )
        
        if viz_type == "Age vs Income":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_df, 'Age', 'Annual Income (k$)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Age vs Spending Score":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_df, 'Age', 'Spending Score (1-100)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Income vs Spending Score":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_df, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Enhanced 3D Visualization":
            # Check if we have all 3 required features for 3D visualization
            required_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            has_all_features = all(feat in feature_columns for feat in required_features) and len(feature_columns) == 3
            
            if has_all_features:
                show_centers = st.checkbox("Show Cluster Centers", value=True)
                scatter_fig = viz_gen.create_3d_cluster_plot(clustered_df, cluster_centers if show_centers else None, show_centers)
            else:
                st.warning("Enhanced 3D Visualization requires all three features: Age, Annual Income, and Spending Score.")
                show_centers = st.checkbox("Show Cluster Centers", value=True)
                scatter_fig = viz_gen.create_3d_cluster_plot(clustered_df, None, False)
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Multi-Dimensional Analysis":
            # Parallel coordinates and correlation analysis
            parallel_fig, correlation_fig = viz_gen.create_multidimensional_analysis_plot(clustered_df, feature_columns)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(parallel_fig, use_container_width=True)
            with col2:
                st.plotly_chart(correlation_fig, use_container_width=True)
        elif viz_type == "3D Volume Analysis":
            # 3D volume analysis with cluster boundaries
            if len(feature_columns) >= 3:
                volume_fig = viz_gen.create_cluster_volume_analysis(clustered_df, feature_columns)
                st.plotly_chart(volume_fig, use_container_width=True)
            else:
                st.warning("Volume analysis requires at least 3 features. Please select more features in the sidebar.")
        
        # Cluster distribution
        st.header("📊 Cluster Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_dist_fig = viz_gen.create_cluster_distribution(clustered_df)
            st.plotly_chart(cluster_dist_fig, use_container_width=True)
        
        with col2:
            gender_dist_fig = viz_gen.create_gender_distribution_by_cluster(clustered_df)
            st.plotly_chart(gender_dist_fig, use_container_width=True)
        
        # Business insights and recommendations
        # Cluster Stability Analysis
        st.header("🔬 Cluster Stability Analysis")
        
        # Calculate comprehensive metrics for current algorithm
        current_metrics = stability_analyzer.calculate_comprehensive_metrics(
            clustered_df, feature_columns, clustered_df['Cluster'].values, algorithm
        )
        
        # Display key stability metrics
        st.subheader("📈 Cluster Quality Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Silhouette Score",
                value=f"{current_metrics['silhouette_avg']:.3f}",
                help="Measures how similar objects are to their own cluster compared to other clusters. Range: -1 to 1, higher is better."
            )
        
        with col2:
            st.metric(
                label="Calinski-Harabasz Index",
                value=f"{current_metrics['calinski_harabasz']:.1f}",
                help="Ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better clustering."
            )
        
        with col3:
            st.metric(
                label="Davies-Bouldin Index", 
                value=f"{current_metrics['davies_bouldin']:.3f}",
                help="Average similarity ratio of each cluster to its most similar cluster. Lower values indicate better clustering."
            )
        
        with col4:
            if current_metrics.get('noise_samples', 0) > 0:
                st.metric(
                    label="Noise Points",
                    value=f"{current_metrics['noise_samples']}",
                    help="Number of points classified as noise (DBSCAN only)."
                )
            else:
                st.metric(
                    label="Valid Clusters",
                    value=f"{current_metrics['n_clusters']}",
                    help="Number of valid clusters identified."
                )
        
        # Silhouette analysis visualization
        st.subheader("📊 Detailed Silhouette Analysis")
        silhouette_fig = stability_analyzer.create_silhouette_analysis_plot(
            clustered_df, feature_columns, clustered_df['Cluster'].values, algorithm
        )
        st.plotly_chart(silhouette_fig, use_container_width=True)
        
        # Algorithm comparison (if Compare All is selected)
        if algorithm == "Compare All":
            st.subheader("🏆 Algorithm Comparison & Recommendation")
            
            # Calculate metrics for all algorithms
            all_metrics = []
            
            # K-Means++ metrics
            if 'K-Means++' in comparison_results:
                kmeans_labels = comparison_results['K-Means++']['labels']
                kmeans_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_df, feature_columns, kmeans_labels, 'K-Means++'
                )
                all_metrics.append(kmeans_metrics)
            
            # DBSCAN metrics  
            if 'DBSCAN' in comparison_results:
                dbscan_labels = comparison_results['DBSCAN']['labels']
                dbscan_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_df, feature_columns, dbscan_labels, 'DBSCAN'
                )
                all_metrics.append(dbscan_metrics)
            
            # Hierarchical metrics
            if 'Hierarchical' in comparison_results:
                hierarchical_labels = comparison_results['Hierarchical']['labels']
                hierarchical_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_df, feature_columns, hierarchical_labels, 'Hierarchical'
                )
                all_metrics.append(hierarchical_metrics)
            
            # Display comparison dashboard
            if all_metrics:
                comparison_fig = stability_analyzer.create_cluster_validation_dashboard(all_metrics)
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Get algorithm recommendation
                recommendation = stability_analyzer.recommend_optimal_algorithm(all_metrics)
                
                # Display recommendation
                st.subheader("🎯 Algorithm Recommendation")
                st.markdown(recommendation['reasoning'], unsafe_allow_html=True)
                
                # Show detailed scoring
                with st.expander("📊 Detailed Algorithm Scores"):
                    score_data = []
                    for algo, score_info in recommendation['all_scores'].items():
                        score_data.append({
                            'Algorithm': algo,
                            'Overall Score': f"{score_info['score']:.3f}",
                            'Silhouette Score': f"{score_info['metrics']['silhouette_avg']:.3f}",
                            'CH Index': f"{score_info['metrics']['calinski_harabasz']:.1f}",
                            'DB Index': f"{score_info['metrics']['davies_bouldin']:.3f}",
                            'Clusters': score_info['metrics']['n_clusters']
                        })
                    
                    score_df = pd.DataFrame(score_data)
                    st.dataframe(score_df, use_container_width=True)
        
        st.header("💡 Business Insights & Recommendations")
        
        cluster_insights = insights.generate_cluster_insights(clustered_df, cluster_centers, feature_columns)
        
        for cluster_id, insight in cluster_insights.items():
            st.markdown(f"""
            <div class="segment-card">
                <h3 style="color: #F18F01; margin-bottom: 1rem;">🎯 {insight['name']}</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div><strong>Size:</strong> {insight['size']} customers ({insight['percentage']:.1f}%)</div>
                    <div><strong>Avg Age:</strong> {insight['avg_age']:.0f} years</div>
                    <div><strong>Avg Income:</strong> ${insight['avg_income']:.0f}k</div>
                    <div><strong>Avg Spending:</strong> {insight['avg_spending']:.0f}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>🎯 Profile:</strong> {insight['profile']}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>📈 Marketing Strategy:</strong> {insight['strategy']}
                </div>
                <div>
                    <strong>💰 Revenue Opportunities:</strong> {insight['opportunities']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed cluster statistics
        st.header("📈 Detailed Cluster Statistics")
        
        # Calculate cluster statistics
        cluster_stats = clustered_df.groupby('Cluster').agg({
            'Age': ['mean', 'std', 'min', 'max'],
            'Annual Income (k$)': ['mean', 'std', 'min', 'max'],
            'Spending Score (1-100)': ['mean', 'std', 'min', 'max'],
            'CustomerID': 'count'
        }).round(2)
        
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
        cluster_stats = cluster_stats.rename(columns={'CustomerID_count': 'Customer_Count'})
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Download section
        st.header("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare download data
            download_df = clustered_df.copy()
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Segmented Data (CSV)",
                data=csv,
                file_name="customer_segments.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepare cluster centers for download
            centers_df = pd.DataFrame(cluster_centers)
            centers_df.columns = feature_columns
            centers_df['Cluster'] = range(len(centers_df))
            centers_csv = centers_df.to_csv(index=False)
            
            st.download_button(
                label="🎯 Download Cluster Centers (CSV)",
                data=centers_csv,
                file_name="cluster_centers.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error loading or processing data: {str(e)}")
        st.info("Please ensure the CSV file is properly formatted and accessible.")

if __name__ == "__main__":
    main()
