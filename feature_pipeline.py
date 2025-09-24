import pandas as pd
import numpy as np
from feature_engineering import FeatureExtractor

class FeaturePipeline:
    """End-to-end feature engineering pipeline"""
    
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.feature_columns = []
    
    def fit_transform(self, live_data: pd.DataFrame, route_id: str = None) -> pd.DataFrame:
        """Extract all features and prepare for modeling"""
        return self.extractor.extract_all_features(live_data, route_id)
    
    def get_feature_importance_ready_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Get data ready for feature importance analysis"""
        
        # Store feature columns for later use
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        # Explicitly drop latitude/longitude from model features per requirement
        self.feature_columns = [
            col for col in numeric_cols 
            if col not in ['latitude', 'longitude', 'next_lat', 'next_lon', 'geometry', 'vehicle_timestamp', 'ts', 'target']
        ]
        # Fill missing values
        numeric_data = data[self.feature_columns].fillna(0)
        
        # Remove infinite values
        numeric_data = numeric_data.replace([np.inf, -np.inf], 0)
        
        return numeric_data
