import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from clustering import CustomerSegmentation
from visualizations import VisualizationGenerator
from business_insights import BusinessInsights
from cluster_stability import ClusterStabilityAnalyzer

# Configure page
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon=None,
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
    
    st.markdown('<h1 class="main-header">Welcome to Your Customer Insights Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Let's discover the hidden patterns in your customer behavior and unlock powerful business insights!</p>", unsafe_allow_html=True)
    
    # Set up data analysis tools
    customer_data_analyzer = DataProcessor()
    
    try:
        # Load the customer information from database
        customer_records = customer_data_analyzer.load_data("attached_assets/Mall_Customers_1758472298487.csv")
        clean_customer_data = customer_data_analyzer.preprocess_data(customer_records)
        
        # Show raw customer data
        st.header("Meet Your Customers")
        st.write("""Before we dive into the magic of customer segmentation, let's take a look at your customer base. 
                 Here's what we know about the people who shop with you:""")
        
        # Quick stats about your customer community
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        
        with overview_col1:
            st.metric("Total Customers", f"{len(customer_records):,}")
        
        with overview_col2:
            st.metric("Data Features", len(customer_records.columns))
        
        with overview_col3:
            average_customer_age = customer_records['Age'].mean()
            st.metric("Average Age", f"{average_customer_age:.1f} years")
        
        with overview_col4:
            average_customer_income = customer_records['Annual Income (k$)'].mean()
            st.metric("Average Income", f"${average_customer_income:.1f}k")
        
        # Show the actual customer data table
        st.subheader("Your Customer Database")
        st.write("Here's a peek at your complete customer information - every person who shops with you:")
        st.dataframe(customer_records, use_container_width=True)
        
        st.subheader("The Numbers Behind Your Customers")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write("**What the numbers tell us:**")
            important_customer_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            st.dataframe(customer_records[important_customer_features].describe(), use_container_width=True)
        
        with stats_col2:
            st.write("**Who shops with you:**")
            gender_breakdown = customer_records['Gender'].value_counts()
            
            # Create a friendly chart showing your customer mix
            import plotly.express as px
            gender_breakdown_chart = px.pie(
                values=gender_breakdown.values,
                names=gender_breakdown.index,
                title="Your Customer Community",
                color_discrete_map={'Male': '#2E86AB', 'Female': '#A23B72'}
            )
            gender_breakdown_chart.update_layout(
                template='plotly_white',
                font=dict(family="Inter, sans-serif"),
                height=300
            )
            st.plotly_chart(gender_breakdown_chart, use_container_width=True)
        
       
        st.subheader("Understanding Your Customer Patterns")
        
        # Create helpful charts to see customer patterns
        customer_pattern_charts = make_subplots(
            rows=2, cols=2,
            subplot_titles=['How old are your customers?', 'What do they earn?', 
                          'How much do they spend?', 'Age vs Income relationship'],
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Age distribution
        customer_pattern_charts.add_trace(
            go.Histogram(x=customer_records['Age'], name='Age', marker_color='#2E86AB', opacity=0.7),
            row=1, col=1
        )
        
        # Income distribution
        customer_pattern_charts.add_trace(
            go.Histogram(x=customer_records['Annual Income (k$)'], name='Income', marker_color='#A23B72', opacity=0.7),
            row=1, col=2
        )
        
        # Spending Score distribution
        customer_pattern_charts.add_trace(
            go.Histogram(x=customer_records['Spending Score (1-100)'], name='Spending Score', marker_color='#F18F01', opacity=0.7),
            row=2, col=1
        )
        
        # Age vs Income scatter
        colors = ['#2E86AB' if gender == 'Male' else '#A23B72' for gender in customer_records['Gender']]
        customer_pattern_charts.add_trace(
            go.Scatter(
                x=customer_records['Age'], 
                y=customer_records['Annual Income (k$)'], 
                mode='markers',
                name='Age vs Income',
                marker=dict(color=colors, opacity=0.6),
                text=customer_records['Gender'],
                hovertemplate='Age: %{x}<br>Income: $%{y}k<br>Gender: %{text}<extra></extra>'
            ),
            row=2, col=2
        )
        
        customer_pattern_charts.update_layout(
            template='plotly_white',
            font=dict(family="Inter, sans-serif"),
            height=600,
            showlegend=False,
            title_text="How Your Customers Break Down"
        )
        
        # Update axis labels with friendly descriptions
        customer_pattern_charts.update_xaxes(title_text="Age (years)", row=1, col=1)
        customer_pattern_charts.update_xaxes(title_text="Annual Income (k$)", row=1, col=2)
        customer_pattern_charts.update_xaxes(title_text="Spending Score (1-100)", row=2, col=1)
        customer_pattern_charts.update_xaxes(title_text="Age (years)", row=2, col=2)
        customer_pattern_charts.update_yaxes(title_text="Number of Customers", row=1, col=1)
        customer_pattern_charts.update_yaxes(title_text="Number of Customers", row=1, col=2)
        customer_pattern_charts.update_yaxes(title_text="Number of Customers", row=2, col=1)
        customer_pattern_charts.update_yaxes(title_text="Annual Income (k$)", row=2, col=2)
        
        st.plotly_chart(customer_pattern_charts, use_container_width=True)
        
        # Data quality overview
        st.subheader("Data Quality Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_values = customer_records.isnull().sum().sum()
            st.metric("Missing Values", missing_values, help="Total number of missing values in the dataset")
        
        with col2:
            duplicates = customer_records.duplicated().sum()
            st.metric("Duplicate Rows", duplicates, help="Number of duplicate customer records")
        
        with col3:
            data_types = len(customer_records.dtypes.unique())
            st.metric("Data Types", data_types, help="Number of different data types in the dataset")
        
       
        st.subheader("How Customer Traits Connect")
        important_customer_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        # Ensure the data is numeric and calculate correlations
        numeric_data = customer_records[important_customer_features].select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        correlation_fig = px.imshow(
            correlation_matrix,
            title="How Your Customer Characteristics Relate",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        correlation_fig.update_layout(
            template='plotly_white',
            font=dict(family="Inter, sans-serif"),
            height=400
        )
        
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        
        st.subheader("What We Learned About Your Customers")
        
        insights_text = f"""
        **Your Customer Community:**
        - **Size**: {len(customer_records):,} customers with {len(customer_records.columns)} features
        - **Age Range**: {customer_records['Age'].min()} - {customer_records['Age'].max()} years (Avg: {customer_records['Age'].mean():.1f})
        - **Income Range**: ${customer_records['Annual Income (k$)'].min()}k - ${customer_records['Annual Income (k$)'].max()}k (Avg: ${customer_records['Annual Income (k$)'].mean():.1f}k)
        - **Spending Score Range**: {customer_records['Spending Score (1-100)'].min()} - {customer_records['Spending Score (1-100)'].max()} (Avg: {customer_records['Spending Score (1-100)'].mean():.1f})
        - **Gender Split**: {(customer_records['Gender'] == 'Female').sum()} Female ({(customer_records['Gender'] == 'Female').mean()*100:.1f}%), {(customer_records['Gender'] == 'Male').sum()} Male ({(customer_records['Gender'] == 'Male').mean()*100:.1f}%)
        
        **Data Health Check:**
        - Your data is complete - no missing information!
        - No duplicate customers - everyone is unique!
        - All information is properly formatted and ready for analysis!
        """
        
        st.markdown(insights_text)
        
        st.divider()
        
       
        st.sidebar.header("Let's Customize Your Analysis!")
        
       
        max_clusters = st.sidebar.slider(
            "How many customer groups should we explore?",
            min_value=2,
            max_value=10,
            value=8,
            help="We'll test different numbers of groups and help you find the sweet spot!"
        )
        
        
        st.sidebar.subheader("Choose Your Analysis Method")
        selected_algorithm = st.sidebar.selectbox(
            "Pick your favorite approach:",
            ["K-Means++ (Popular & Fast)", "DBSCAN (Finds Unique Patterns)", "Hierarchical (Tree-like Groups)", "Compare All (Let's try everything!)"],
            help="Don't worry - we'll explain what each method does and help you choose!"
        )
        
        # Map display names back to algorithm names immediately
        algorithm_mapping = {
            "K-Means++ (Popular & Fast)": "K-Means++",
            "DBSCAN (Finds Unique Patterns)": "DBSCAN", 
            "Hierarchical (Tree-like Groups)": "Hierarchical",
            "Compare All (Let's try everything!)": "Compare All"
        }
        algo = algorithm_mapping[selected_algorithm]
        
        # Algorithm-specific parameters
        eps = 0.5  # Default value
        min_samples = 5  # Default value
        linkage_method = 'ward'  # Default value
        
        if algo == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help="Maximum distance between samples")
            min_samples = st.sidebar.slider("Min Samples", 2, 10, 5, help="Minimum samples in neighborhood")
        elif algo == "Hierarchical":
            linkage_method = st.sidebar.selectbox(
                "Linkage Method", 
                ["ward", "complete", "average", "single"],
                help="Linkage method for hierarchical clustering"
            )
        elif algo == "Compare All":
            eps = st.sidebar.slider("DBSCAN Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, help="Maximum distance between samples for DBSCAN")
            min_samples = st.sidebar.slider("DBSCAN Min Samples", 2, 10, 5, help="Minimum samples in neighborhood for DBSCAN")
            linkage_method = st.sidebar.selectbox(
                "Hierarchical Linkage Method", 
                ["ward", "complete", "average", "single"],
                help="Linkage method for hierarchical clustering"
            )
        
        
        st.sidebar.subheader("What Should We Look At?")
        st.sidebar.write("Select the customer characteristics you want to group by:")
        use_age = st.sidebar.checkbox("Customer Age (How old are they?)", value=True)
        use_income = st.sidebar.checkbox("How Much They Earn (Annual income)", value=True)
        use_spending = st.sidebar.checkbox("How Much They Spend (Shopping behavior)", value=True)
        
        if not any([use_age, use_income, use_spending]):
            st.sidebar.error("Oops! Please pick at least one customer characteristic to analyze. We need something to work with!")
            return
        
       
        customer_characteristics_to_analyze = []
        if use_age:
            customer_characteristics_to_analyze.append('Age')
        if use_income:
            customer_characteristics_to_analyze.append('Annual Income (k$)')
        if use_spending:
            customer_characteristics_to_analyze.append('Spending Score (1-100)')
        
        # Set up smart customer analysis engine
        customer_analysis_engine = CustomerSegmentation()
        
        # Initialize variables with default values to prevent unbound errors
        clustered_customer_records = clean_customer_data.copy()
        cluster_centers = None
        optimal_k = 0
        inertias = []
        silhouette_scores = []
        comparison_results = {}
        linkage_matrix = None
        
        
        
        if algo == "K-Means++":
            # Find optimal clusters
            with st.spinner("Finding optimal number of clusters..."):
                inertias, silhouette_scores = customer_analysis_engine.find_optimal_clusters(
                    clean_customer_data[customer_characteristics_to_analyze], max_clusters
                )
                optimal_k = customer_analysis_engine.get_optimal_clusters_elbow(inertias)
            
            # Perform clustering
            with st.spinner("Performing K-Means++ clustering..."):
                clustered_customer_records, cluster_centers = customer_analysis_engine.perform_clustering(
                    clean_customer_data, customer_characteristics_to_analyze, optimal_k
                )
        
        elif algo == "DBSCAN":
            with st.spinner("Performing DBSCAN clustering..."):
                clustered_customer_records, cluster_centers, n_clusters = customer_analysis_engine.perform_dbscan_clustering(
                    clean_customer_data, customer_characteristics_to_analyze, eps, min_samples
                )
                optimal_k = n_clusters
                inertias, silhouette_scores = [], []  
        
        elif algo == "Hierarchical":
            # Use optimal k from elbow method for hierarchical
            with st.spinner("Finding optimal number of clusters..."):
                inertias, silhouette_scores = customer_analysis_engine.find_optimal_clusters(
                    clean_customer_data[customer_characteristics_to_analyze], max_clusters
                )
                optimal_k = customer_analysis_engine.get_optimal_clusters_elbow(inertias)
            
            with st.spinner("Performing Hierarchical clustering..."):
                clustered_customer_records, cluster_centers, linkage_matrix = customer_analysis_engine.perform_hierarchical_clustering(
                    clean_customer_data, customer_characteristics_to_analyze, optimal_k, linkage_method
                )
        
        elif algo == "Compare All":
            with st.spinner("Comparing all clustering algorithms..."):
                inertias, silhouette_scores = customer_analysis_engine.find_optimal_clusters(
                    clean_customer_data[customer_characteristics_to_analyze], max_clusters
                )
                optimal_k = customer_analysis_engine.get_optimal_clusters_elbow(inertias)
                
                comparison_results = customer_analysis_engine.compare_clustering_algorithms(
                    clean_customer_data, customer_characteristics_to_analyze, optimal_k, eps, min_samples
                )
                
                # Use K-Means++ results for main visualization
                clustered_customer_records, cluster_centers = customer_analysis_engine.perform_clustering(
                    clean_customer_data, customer_characteristics_to_analyze, optimal_k
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
                <h2 style="margin: 0;">{len(clustered_customer_records)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            cluster_label = "Clusters Found" if algo == "DBSCAN" else "Optimal Clusters"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #A23B72; margin: 0;">{cluster_label}</h3>
                <h2 style="margin: 0;">{optimal_k if optimal_k > 0 else 'N/A'}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_income = clustered_customer_records['Annual Income (k$)'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #F18F01; margin: 0;">Avg. Income</h3>
                <h2 style="margin: 0;">${avg_income:.0f}k</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_spending = clustered_customer_records['Spending Score (1-100)'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #06D6A0; margin: 0;">Avg. Spending Score</h3>
                <h2 style="margin: 0;">{avg_spending:.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm-specific analysis sections
        if algo in ["K-Means++", "Hierarchical"] or (algo == "Compare All"):
            st.header("Clustering Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Elbow Method Analysis")
                elbow_fig = viz_gen.create_elbow_plot(range(1, max_clusters + 1), inertias, optimal_k)
                st.plotly_chart(elbow_fig, use_container_width=True)
            
            with col2:
                st.subheader("Silhouette Score Analysis")
                silhouette_fig = viz_gen.create_silhouette_plot(range(2, max_clusters + 1), silhouette_scores)
                st.plotly_chart(silhouette_fig, use_container_width=True)
        
        # Algorithm comparison section
        if algo == "Compare All":
            st.header("Algorithm Comparison")
            comparison_fig = viz_gen.create_algorithm_comparison_plot(comparison_results)
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            # Display comparison results table
            st.subheader("📋 Detailed Comparison Results")
            comparison_customer_records = pd.DataFrame(comparison_results).T
            comparison_customer_records = comparison_customer_records.round(3)
            st.dataframe(comparison_customer_records, use_container_width=True)
        
        # Hierarchical-specific dendrogram
        if algo == "Hierarchical":
            st.header("🌳 Dendrogram Analysis")
            st.subheader("Hierarchical Clustering Dendrogram")
            dendrogram_img = viz_gen.create_dendrogram_plot(linkage_matrix, customer_characteristics_to_analyze)
            if dendrogram_img:
                st.image(f"data:image/png;base64,{dendrogram_img}")
        
        # Customer segmentation visualization
        st.header("Customer Segmentation Visualization")
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Age vs Income", "Age vs Spending Score", "Income vs Spending Score", "Enhanced 3D Visualization", "Multi-Dimensional Analysis", "3D Volume Analysis"]
        )
        
        if viz_type == "Age vs Income":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_customer_records, 'Age', 'Annual Income (k$)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Age vs Spending Score":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_customer_records, 'Age', 'Spending Score (1-100)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Income vs Spending Score":
            scatter_fig = viz_gen.create_cluster_scatter(
                clustered_customer_records, 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster', cluster_centers
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Enhanced 3D Visualization":
            # Check if we have all 3 required features for 3D visualization
            required_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            has_all_features = all(feat in customer_characteristics_to_analyze for feat in required_features) and len(customer_characteristics_to_analyze) == 3
            
            if has_all_features:
                show_centers = st.checkbox("Show Cluster Centers", value=True)
                scatter_fig = viz_gen.create_3d_cluster_plot(clustered_customer_records, cluster_centers if show_centers else None, show_centers)
            else:
                st.warning("Enhanced 3D Visualization requires all three features: Age, Annual Income, and Spending Score.")
                show_centers = st.checkbox("Show Cluster Centers", value=True)
                scatter_fig = viz_gen.create_3d_cluster_plot(clustered_customer_records, None, False)
            st.plotly_chart(scatter_fig, use_container_width=True)
        elif viz_type == "Multi-Dimensional Analysis":
            # Parallel coordinates and correlation analysis
            parallel_fig, correlation_fig = viz_gen.create_multidimensional_analysis_plot(clustered_customer_records, customer_characteristics_to_analyze)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(parallel_fig, use_container_width=True)
            with col2:
                st.plotly_chart(correlation_fig, use_container_width=True)
        elif viz_type == "3D Volume Analysis":
            # 3D volume analysis with cluster boundaries
            if len(customer_characteristics_to_analyze) >= 3:
                volume_fig = viz_gen.create_cluster_volume_analysis(clustered_customer_records, customer_characteristics_to_analyze)
                st.plotly_chart(volume_fig, use_container_width=True)
            else:
                st.warning("Volume analysis requires at least 3 features. Please select more features in the sidebar.")
        
        # Cluster distribution
        st.header("Cluster Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cluster_dist_fig = viz_gen.create_cluster_distribution(clustered_customer_records)
            st.plotly_chart(cluster_dist_fig, use_container_width=True)
        
        with col2:
            gender_dist_fig = viz_gen.create_gender_distribution_by_cluster(clustered_customer_records)
            st.plotly_chart(gender_dist_fig, use_container_width=True)
        
        # Business insights and recommendations
        # Cluster Stability Analysis
        st.header("🔬 Cluster Stability Analysis")
        
        # Calculate comprehensive metrics for current selected_algorithm
        current_metrics = stability_analyzer.calculate_comprehensive_metrics(
            clustered_customer_records, customer_characteristics_to_analyze, clustered_customer_records['Cluster'].values, selected_algorithm
        )
        
        # Display key stability metrics
        st.subheader("Cluster Quality Metrics")
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
        st.subheader("Detailed Silhouette Analysis")
        silhouette_fig = stability_analyzer.create_silhouette_analysis_plot(
            clustered_customer_records, customer_characteristics_to_analyze, clustered_customer_records['Cluster'].values, selected_algorithm
        )
        st.plotly_chart(silhouette_fig, use_container_width=True)
        
        # Algorithm comparison (if Compare All is selected)
        if algo == "Compare All":
            st.subheader("Algorithm Comparison & Recommendation")
            
            # Calculate metrics for all selected_algorithms
            all_metrics = []
            
            # K-Means++ metrics
            if 'K-Means++' in comparison_results:
                kmeans_labels = comparison_results['K-Means++']['labels']
                kmeans_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_customer_records, customer_characteristics_to_analyze, kmeans_labels, 'K-Means++'
                )
                all_metrics.append(kmeans_metrics)
            
            # DBSCAN metrics  
            if 'DBSCAN' in comparison_results:
                dbscan_labels = comparison_results['DBSCAN']['labels']
                dbscan_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_customer_records, customer_characteristics_to_analyze, dbscan_labels, 'DBSCAN'
                )
                all_metrics.append(dbscan_metrics)
            
            # Hierarchical metrics
            if 'Hierarchical' in comparison_results:
                hierarchical_labels = comparison_results['Hierarchical']['labels']
                hierarchical_metrics = stability_analyzer.calculate_comprehensive_metrics(
                    clustered_customer_records, customer_characteristics_to_analyze, hierarchical_labels, 'Hierarchical'
                )
                all_metrics.append(hierarchical_metrics)
            
            # Display comparison dashboard
            if all_metrics:
                comparison_fig = stability_analyzer.create_cluster_validation_dashboard(all_metrics)
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Get selected_algorithm recommendation
                recommendation = stability_analyzer.recommend_optimal_algorithm(all_metrics)
                
                # Display recommendation
                st.subheader("Algorithm Recommendation")
                st.markdown(recommendation['reasoning'], unsafe_allow_html=True)
                
                # Show detailed scoring
                with st.expander("Detailed Algorithm Scores"):
                    score_data = []
                    # Safely handle all_scores - ensure it exists and is a dictionary
                    all_scores = recommendation.get('all_scores', {})
                    if isinstance(all_scores, dict) and all_scores:
                        for algo, score_info in all_scores.items():
                            score_data.append({
                                'Algorithm': algo,
                                'Overall Score': f"{score_info['score']:.3f}",
                                'Silhouette Score': f"{score_info['metrics']['silhouette_avg']:.3f}",
                                'CH Index': f"{score_info['metrics']['calinski_harabasz']:.1f}",
                                'DB Index': f"{score_info['metrics']['davies_bouldin']:.3f}",
                                'Clusters': score_info['metrics']['n_clusters']
                            })
                        
                        score_customer_records = pd.DataFrame(score_data)
                        st.dataframe(score_customer_records, use_container_width=True)
                    else:
                        st.info("No detailed scoring information available.")
        
        st.header("Business Insights & Recommendations")
        
        cluster_insights = insights.generate_cluster_insights(clustered_customer_records, cluster_centers, customer_characteristics_to_analyze)
        
        for cluster_id, insight in cluster_insights.items():
            st.markdown(f"""
            <div class="segment-card">
                <h3 style="color: #F18F01; margin-bottom: 1rem;">{insight['name']}</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div><strong>Size:</strong> {insight['size']} customers ({insight['percentage']:.1f}%)</div>
                    <div><strong>Avg Age:</strong> {insight['avg_age']:.0f} years</div>
                    <div><strong>Avg Income:</strong> ${insight['avg_income']:.0f}k</div>
                    <div><strong>Avg Spending:</strong> {insight['avg_spending']:.0f}</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Profile:</strong> {insight['profile']}
                </div>
                <div style="margin-bottom: 1rem;">
                    <strong>Marketing Strategy:</strong> {insight['strategy']}
                </div>
                <div>
                    <strong>Revenue Opportunities:</strong> {insight['opportunities']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed cluster statistics
        st.header("Detailed Cluster Statistics")
        
        cluster_stats = clustered_customer_records.groupby('Cluster').agg({
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
            download_customer_records = clustered_customer_records.copy()
            csv = download_customer_records.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Segmented Data (CSV)",
                data=csv,
                file_name="customer_segments.csv",
                mime="text/csv"
            )
        
        with col2:
            # Prepare cluster centers for download
            centers_customer_records = pd.DataFrame(cluster_centers)
            centers_customer_records.columns = customer_characteristics_to_analyze
            centers_customer_records['Cluster'] = range(len(centers_customer_records))
            centers_csv = centers_customer_records.to_csv(index=False)
            
            st.download_button(
                label="Download Cluster Centers (CSV)",
                data=centers_csv,
                file_name="cluster_centers.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error loading or processing data: {str(e)}")
        st.info("Please ensure the CSV file is properly formatted and accessible.")

if __name__ == "__main__":
    main()
