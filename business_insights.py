import pandas as pd
import numpy as np
import streamlit as st

class BusinessInsights:
    """
    Generates business insights and recommendations based on customer segmentation.
    """
    
    def __init__(self):
        # Define customer personas and marketing strategies
        self.personas = {
            'high_income_high_spending': {
                'name': 'Premium Customers',
                'description': 'High-income customers with high spending scores',
                'strategy': 'Premium product offerings, VIP services, exclusive events',
                'opportunities': 'Luxury goods, premium services, personalized experiences'
            },
            'high_income_low_spending': {
                'name': 'Potential Customers',
                'description': 'High-income customers with low spending scores',
                'strategy': 'Targeted promotions, loyalty programs, value proposition communication',
                'opportunities': 'Convert to high spenders through targeted campaigns'
            },
            'low_income_high_spending': {
                'name': 'Loyal Enthusiasts',
                'description': 'Lower-income customers with high spending scores',
                'strategy': 'Affordable options, payment plans, customer retention programs',
                'opportunities': 'Budget-friendly product lines, financing options'
            },
            'low_income_low_spending': {
                'name': 'Budget Conscious',
                'description': 'Lower-income customers with low spending scores',
                'strategy': 'Value pricing, promotional offers, basic product lines',
                'opportunities': 'Entry-level products, seasonal promotions'
            },
            'middle_segment': {
                'name': 'Mainstream Customers',
                'description': 'Average income and spending customers',
                'strategy': 'Balanced product mix, regular promotions, standard service',
                'opportunities': 'Core product offerings, cross-selling opportunities'
            }
        }
    
    def generate_cluster_insights(self, clustered_df, cluster_centers, feature_columns):
        """
        Generate comprehensive business insights for each cluster.
        
        Args:
            clustered_df (pd.DataFrame): Customer data with cluster assignments
            cluster_centers (np.ndarray): Cluster center coordinates
            feature_columns (list): Features used for clustering
            
        Returns:
            dict: Detailed insights for each cluster
        """
        try:
            insights = {}
            
            # Calculate overall statistics for comparison
            overall_income = clustered_df['Annual Income (k$)'].mean()
            overall_spending = clustered_df['Spending Score (1-100)'].mean()
            overall_age = clustered_df['Age'].mean()
            
            for cluster_id in sorted(clustered_df['Cluster'].unique()):
                cluster_data = clustered_df[clustered_df['Cluster'] == cluster_id]
                
                # Basic cluster statistics
                avg_age = cluster_data['Age'].mean()
                avg_income = cluster_data['Annual Income (k$)'].mean()
                avg_spending = cluster_data['Spending Score (1-100)'].mean()
                size = len(cluster_data)
                percentage = (size / len(clustered_df)) * 100
                
                # Gender distribution
                gender_dist = cluster_data['Gender'].value_counts(normalize=True) * 100
                dominant_gender = gender_dist.index[0]
                
                # Classify cluster type
                cluster_type = self._classify_cluster(avg_income, avg_spending, overall_income, overall_spending)
                persona = self.personas.get(cluster_type, self.personas['middle_segment'])
                
                # Generate detailed profile
                profile = self._generate_cluster_profile(
                    avg_age, avg_income, avg_spending, gender_dist, overall_age, overall_income, overall_spending
                )
                
                # Generate marketing strategy
                strategy = self._generate_marketing_strategy(cluster_type, avg_age, avg_income, avg_spending, size)
                
                # Generate revenue opportunities
                opportunities = self._generate_revenue_opportunities(cluster_type, avg_age, avg_income, avg_spending, size)
                
                insights[cluster_id] = {
                    'name': persona['name'],
                    'size': size,
                    'percentage': percentage,
                    'avg_age': avg_age,
                    'avg_income': avg_income,
                    'avg_spending': avg_spending,
                    'dominant_gender': dominant_gender,
                    'gender_distribution': gender_dist.to_dict(),
                    'cluster_type': cluster_type,
                    'profile': profile,
                    'strategy': strategy,
                    'opportunities': opportunities,
                    'priority_level': self._calculate_priority_level(avg_income, avg_spending, size, percentage)
                }
            
            return insights
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return {}
    
    def _classify_cluster(self, avg_income, avg_spending, overall_income, overall_spending):
        """
        Classify cluster based on income and spending patterns.
        
        Args:
            avg_income (float): Average income of cluster
            avg_spending (float): Average spending score of cluster
            overall_income (float): Overall average income
            overall_spending (float): Overall average spending
            
        Returns:
            str: Cluster classification
        """
        income_threshold = overall_income
        spending_threshold = overall_spending
        
        if avg_income > income_threshold and avg_spending > spending_threshold:
            return 'high_income_high_spending'
        elif avg_income > income_threshold and avg_spending <= spending_threshold:
            return 'high_income_low_spending'
        elif avg_income <= income_threshold and avg_spending > spending_threshold:
            return 'low_income_high_spending'
        elif avg_income <= income_threshold and avg_spending <= spending_threshold:
            return 'low_income_low_spending'
        else:
            return 'middle_segment'
    
    def _generate_cluster_profile(self, avg_age, avg_income, avg_spending, gender_dist, 
                                 overall_age, overall_income, overall_spending):
        """
        Generate detailed cluster profile description.
        
        Args:
            avg_age (float): Average age
            avg_income (float): Average income
            avg_spending (float): Average spending
            gender_dist (pd.Series): Gender distribution
            overall_age (float): Overall average age
            overall_income (float): Overall average income
            overall_spending (float): Overall average spending
            
        Returns:
            str: Detailed profile description
        """
        age_desc = "young" if avg_age < overall_age - 5 else "older" if avg_age > overall_age + 5 else "middle-aged"
        income_desc = "high-income" if avg_income > overall_income * 1.2 else "low-income" if avg_income < overall_income * 0.8 else "moderate-income"
        spending_desc = "high-spending" if avg_spending > overall_spending * 1.2 else "low-spending" if avg_spending < overall_spending * 0.8 else "moderate-spending"
        
        dominant_gender = gender_dist.index[0].lower()
        gender_proportion = gender_dist.iloc[0]
        
        gender_desc = f"predominantly {dominant_gender}" if gender_proportion > 70 else f"slightly more {dominant_gender}" if gender_proportion > 60 else "gender-balanced"
        
        profile = f"This segment consists of {age_desc}, {income_desc}, {spending_desc} customers who are {gender_desc}. "
        profile += f"They represent a key demographic with an average age of {avg_age:.0f} years, "
        profile += f"annual income of ${avg_income:.0f}k, and spending score of {avg_spending:.0f}."
        
        return profile
    
    def _generate_marketing_strategy(self, cluster_type, avg_age, avg_income, avg_spending, size):
        """
        Generate targeted marketing strategy for the cluster.
        
        Args:
            cluster_type (str): Type of cluster
            avg_age (float): Average age
            avg_income (float): Average income
            avg_spending (float): Average spending
            size (int): Cluster size
            
        Returns:
            str: Marketing strategy recommendation
        """
        base_strategy = self.personas[cluster_type]['strategy']
        
        # Age-based modifications
        if avg_age < 30:
            age_strategy = "Focus on digital marketing, social media campaigns, and trendy products. "
        elif avg_age > 50:
            age_strategy = "Emphasize traditional marketing channels, quality, and reliability. "
        else:
            age_strategy = "Use mixed marketing channels with emphasis on value and convenience. "
        
        # Size-based modifications
        if size > 50:
            size_strategy = "This is a large segment warranting significant marketing investment. "
        elif size < 20:
            size_strategy = "This is a niche segment suitable for specialized, targeted campaigns. "
        else:
            size_strategy = "This is a moderate-sized segment with good potential for growth. "
        
        return f"{base_strategy} {age_strategy}{size_strategy}"
    
    def _generate_revenue_opportunities(self, cluster_type, avg_age, avg_income, avg_spending, size):
        """
        Generate revenue opportunity recommendations.
        
        Args:
            cluster_type (str): Type of cluster
            avg_age (float): Average age
            avg_income (float): Average income
            avg_spending (float): Average spending
            size (int): Cluster size
            
        Returns:
            str: Revenue opportunities description
        """
        base_opportunities = self.personas[cluster_type]['opportunities']
        
        # Calculate potential revenue impact
        potential_revenue = size * avg_income * (avg_spending / 100) * 0.1  # Rough estimate
        
        revenue_desc = f"Estimated revenue potential: ${potential_revenue:.0f}k annually. "
        
        # Age-specific opportunities
        if avg_age < 30:
            age_opportunities = "Target technology products, fashion, entertainment, and experience-based services. "
        elif avg_age > 50:
            age_opportunities = "Focus on health & wellness, home improvement, travel, and premium services. "
        else:
            age_opportunities = "Emphasize family-oriented products, convenience services, and lifestyle brands. "
        
        # Income and spending specific opportunities
        if avg_income > 70 and avg_spending > 70:
            specific_opportunities = "Premium product lines, exclusive memberships, and luxury experiences. "
        elif avg_income < 40 and avg_spending < 40:
            specific_opportunities = "Value products, bulk discounts, and affordable alternatives. "
        else:
            specific_opportunities = "Mid-range products with good value proposition and flexible pricing. "
        
        return f"{base_opportunities} {revenue_desc}{age_opportunities}{specific_opportunities}"
    
    def _calculate_priority_level(self, avg_income, avg_spending, size, percentage):
        """
        Calculate priority level for business focus.
        
        Args:
            avg_income (float): Average income
            avg_spending (float): Average spending
            size (int): Cluster size
            percentage (float): Percentage of total customers
            
        Returns:
            str: Priority level (High, Medium, Low)
        """
        # Calculate composite score
        income_score = min(avg_income / 100, 1.0)  # Normalize to 0-1
        spending_score = avg_spending / 100  # Already 0-1
        size_score = min(percentage / 30, 1.0)  # Normalize to 0-1 (30% = max)
        
        composite_score = (income_score * 0.4 + spending_score * 0.4 + size_score * 0.2)
        
        if composite_score > 0.7:
            return "High"
        elif composite_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def generate_overall_recommendations(self, insights):
        """
        Generate overall business recommendations based on all clusters.
        
        Args:
            insights (dict): Cluster insights
            
        Returns:
            dict: Overall recommendations
        """
        try:
            total_customers = sum(insight['size'] for insight in insights.values())
            high_priority_clusters = [k for k, v in insights.items() if v['priority_level'] == 'High']
            
            recommendations = {
                'key_findings': [],
                'strategic_focus': [],
                'resource_allocation': [],
                'growth_opportunities': []
            }
            
            # Key findings
            largest_cluster = max(insights.items(), key=lambda x: x[1]['size'])
            most_valuable_cluster = max(insights.items(), key=lambda x: x[1]['avg_income'] * x[1]['avg_spending'])
            
            recommendations['key_findings'].append(
                f"Largest segment: {largest_cluster[1]['name']} ({largest_cluster[1]['percentage']:.1f}% of customers)"
            )
            recommendations['key_findings'].append(
                f"Most valuable segment: {most_valuable_cluster[1]['name']} (highest income-spending potential)"
            )
            recommendations['key_findings'].append(
                f"Total customers analyzed: {total_customers:,}"
            )
            recommendations['key_findings'].append(
                f"High priority segments: {len(high_priority_clusters)} out of {len(insights)} total segments"
            )
            
            # Strategic focus recommendations
            for cluster_id in high_priority_clusters[:2]:  # Top 2 priority clusters
                cluster = insights[cluster_id]
                recommendations['strategic_focus'].append(
                    f"Focus on {cluster['name']}: {cluster['strategy'][:100]}..."
                )
            
            # Resource allocation
            high_value_segments = [k for k, v in insights.items() if v['avg_income'] > 60 and v['avg_spending'] > 60]
            growth_segments = [k for k, v in insights.items() if v['avg_income'] > 60 and v['avg_spending'] < 50]
            
            if high_value_segments:
                recommendations['resource_allocation'].append(
                    f"Allocate 40-50% of marketing budget to high-value segments (Clusters: {high_value_segments})"
                )
            
            if growth_segments:
                recommendations['resource_allocation'].append(
                    f"Invest 20-30% in growth potential segments (Clusters: {growth_segments})"
                )
            
            # Growth opportunities
            for cluster_id, cluster in insights.items():
                if cluster['priority_level'] == 'High':
                    recommendations['growth_opportunities'].append(
                        f"{cluster['name']}: {cluster['opportunities'][:100]}..."
                    )
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error generating overall recommendations: {str(e)}")
            return {}
