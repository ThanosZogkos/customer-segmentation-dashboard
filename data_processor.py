import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    
   # Handles data loading, cleaning, and preprocessing for customer segmentation analysis.
  
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.original_data = None
        self.processed_data = None
    
    def load_data(self, file_path):
       
       # Load customer data from CSV file.
        
        try:
            df = pd.read_csv(file_path)
            self.original_data = df.copy()
            
            # Basic data validation
            required_columns = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            st.success(f"Successfully loaded {len(df)} customer records")
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise e
    
    def preprocess_data(self, df):
      
        #Clean and preprocess the customer data.
       
        try:
            # Create a copy for processing
            processed_df = df.copy()
            
            # Data quality checks and cleaning
            initial_count = len(processed_df)
            
            # Remove duplicates
            processed_df = processed_df.drop_duplicates()
            duplicates_removed = initial_count - len(processed_df)
            
            # Handle missing values
            missing_before = processed_df.isnull().sum().sum()
            processed_df = processed_df.dropna()
            missing_removed = missing_before
            
            # Data type validation and conversion
            processed_df['CustomerID'] = processed_df['CustomerID'].astype(int)
            processed_df['Age'] = pd.to_numeric(processed_df['Age'], errors='coerce')
            processed_df['Annual Income (k$)'] = pd.to_numeric(processed_df['Annual Income (k$)'], errors='coerce')
            processed_df['Spending Score (1-100)'] = pd.to_numeric(processed_df['Spending Score (1-100)'], errors='coerce')
            
            # Remove any rows with invalid numeric data
            processed_df = processed_df.dropna()
            
            # Data validation - check for reasonable ranges
            processed_df = processed_df[
                (processed_df['Age'] >= 15) & (processed_df['Age'] <= 100) &
                (processed_df['Annual Income (k$)'] >= 0) & (processed_df['Annual Income (k$)'] <= 200) &
                (processed_df['Spending Score (1-100)'] >= 1) & (processed_df['Spending Score (1-100)'] <= 100)
            ]
             
            # Standardize gender values
            processed_df['Gender'] = processed_df['Gender'].str.strip().str.title()
            
            outliers_removed = len(df) - len(processed_df) - duplicates_removed - missing_removed
            
            # Display preprocessing summary
            if duplicates_removed > 0 or missing_removed > 0 or outliers_removed > 0:
                st.info(f"""
                **Data Preprocessing Summary:**
                - Original records: {initial_count}
                - Duplicates removed: {duplicates_removed}
                - Missing values removed: {missing_removed}
                - Outliers removed: {outliers_removed}
                - **Final records: {len(processed_df)}**
                """)
            
            # Store processed data
            self.processed_data = processed_df
            
            # Display basic statistics
            self.display_data_summary(processed_df)
            
            return processed_df
            
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            raise e
    
    def display_data_summary(self, df):
       
        Display summary statistics of the processed data.
        
        try:
            st.subheader("Data Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numerical Statistics**")
                numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
                st.dataframe(df[numeric_cols].describe())
            
            with col2:
                st.write("**Gender Distribution**")
                gender_dist = df['Gender'].value_counts()
                st.dataframe(gender_dist.to_frame('Count'))
            
            # Age distribution insights
            age_groups = pd.cut(df['Age'], 
                              bins=[0, 25, 35, 50, 65, 100], 
                              labels=['18-25', '26-35', '36-50', '51-65', '65+'])
            age_dist = pd.Series(age_groups).value_counts().sort_index()
            
            st.write("**Age Group Distribution**")
            st.dataframe(age_dist.to_frame('Count'))
            
        except Exception as e:
            st.warning(f"Could not display data summary: {str(e)}")
    
    def scale_features(self, df, feature_columns):
        
       # Scale the selected features for clustering.
        
        try:
            features = df[feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            st.success(f"Successfully scaled {len(feature_columns)} features")
            return scaled_features
            
        except Exception as e:
            st.error(f"Error scaling features: {str(e)}")
            raise e
    
    def get_feature_importance(self, df):
       
       # Calculate basic feature statistics for importance analysis.
        
        try:
            numeric_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            importance = {}
            
            for col in numeric_cols:
                importance[col] = {
                    'variance': df[col].var(),
                    'range': df[col].max() - df[col].min(),
                    'coefficient_of_variation': df[col].std() / df[col].mean()
                }
            
            return importance
            
        except Exception as e:
            st.warning(f"Could not calculate feature importance: {str(e)}")
            return {}
    
    def detect_outliers(self, df, column, method='iqr'):
        
       # Detect outliers in a specific column.
    
        try:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > 3
            
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")
            
            return outliers
            
        except Exception as e:
            st.warning(f"Could not detect outliers: {str(e)}")
            return pd.Series([False] * len(df))
