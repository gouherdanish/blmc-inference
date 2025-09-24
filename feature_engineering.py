"""
Feature Engineering for Bus Arrival Prediction
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import geopandas as gpd
from geoutils import GeoUtils
import pandas as pd

class FeatureExtractor:
    """Extract features for bus arrival prediction"""
    
    def __init__(self):
        self.route_characteristics = {}
        
    def extract_temporal_features(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive temporal features"""
        
        # Basic temporal features
        live_data['hour'] = live_data['ts'].dt.hour
        live_data['day_of_week'] = live_data['ts'].dt.dayofweek
        live_data['day_of_month'] = live_data['ts'].dt.day
        live_data['month'] = live_data['ts'].dt.month
        live_data['week_of_year'] = live_data['ts'].dt.isocalendar().week
        
        # Advanced temporal patterns
        live_data['is_weekend'] = live_data['day_of_week'].isin([5, 6])
        live_data['is_rush_hour'] = live_data['hour'].isin([7, 8, 9, 17, 18, 19])
        live_data['is_morning_peak'] = live_data['hour'].isin([7, 8, 9])
        live_data['is_evening_peak'] = live_data['hour'].isin([17, 18, 19])
        live_data['is_lunch_hour'] = live_data['hour'].isin([12, 13, 14])
        live_data['is_night'] = live_data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])
        
        # Cyclical encoding for time features
        live_data['hour_sin'] = np.sin(2 * np.pi * live_data['hour'] / 24)
        live_data['hour_cos'] = np.cos(2 * np.pi * live_data['hour'] / 24)
        live_data['day_sin'] = np.sin(2 * np.pi * live_data['day_of_week'] / 7)
        live_data['day_cos'] = np.cos(2 * np.pi * live_data['day_of_week'] / 7)
        live_data['month_sin'] = np.sin(2 * np.pi * live_data['month'] / 12)
        live_data['month_cos'] = np.cos(2 * np.pi * live_data['month'] / 12)
        
        return live_data
    
    def extract_spatial_features(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Extract spatial and route-based features"""
        live_data = self._calculate_segment_metrics(live_data)
        live_data = self._calculate_movement_patterns(live_data)
        return live_data
    
    def extract_route_characteristics(self, live_data: pd.DataFrame, 
                                     route_id: str) -> pd.DataFrame:
        """Extract route-specific characteristics"""
        
        if route_id not in self.route_characteristics:
            self._compute_route_characteristics(live_data, route_id)
        
        route_chars = self.route_characteristics[route_id]
        
        # Add route characteristics to trip data
        live_data['route_length'] = route_chars['avg_trip_length']
        live_data['route_complexity'] = route_chars['complexity_score']
        live_data['route_stop_density'] = route_chars['stop_density']
        live_data['route_avg_speed'] = route_chars['avg_speed']
        
        # Stop sequence features using basic route context from data processor
        live_data['stops_remaining'] = live_data.get('total_route_stops', 10) - live_data.get('current_position_in_route', 0)
        live_data['progress_ratio'] = live_data.get('current_position_in_route', 0) / live_data.get('total_route_stops', 10)
        
        return live_data
    
    def extract_all_features(self, live_data: pd.DataFrame, 
                            route_id: str = None) -> pd.DataFrame:
        """Extract all features in one go"""
        
        print("Extracting temporal features...")
        live_data = self.extract_temporal_features(live_data)
        
        print("Extracting spatial features...")
        live_data = self.extract_spatial_features(live_data)
        
        if route_id:
            print(f"Extracting route characteristics for {route_id}...")
            live_data = self.extract_route_characteristics(live_data, route_id)
        
        return live_data
    
    # Helper methods
    def _calculate_segment_metrics(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental segment metrics: distance, duration, speed - FULLY VECTORIZED"""
        
        # No null checking needed! Data processor fills nulls with same values
        # Last point of each trip: same->same = 0 distance, 0 duration naturally
        
        # Create geometries for current and next points
        current_geometries = live_data['geometry']
        next_geometries = gpd.points_from_xy(live_data['next_lon'], live_data['next_lat'])
        
        # Calculate distances vectorized
        live_data['segment_distance'] = GeoUtils.calculate_dist_vectorized(current_geometries, next_geometries)
        
        # Calculate segment duration in seconds
        durations = (live_data['next_timestamp'] - live_data['ts']).dt.total_seconds()
        live_data['segment_duration'] = durations.clip(lower=0)  # Vectorized max(0, duration)
        live_data['next_timestamp'] = live_data['next_timestamp'].fillna(live_data['ts'])
        
        return live_data
    
    def _calculate_movement_patterns(self, live_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement patterns"""
        
        # Calculate bearing change
        live_data['bearing_change'] = live_data.groupby('trip_id')['bearing'].diff(-1)
        live_data['bearing_change'] = live_data['bearing_change'].fillna(0)
        live_data['bearing_change_per_meter'] = np.where(
            live_data['segment_distance'] > 0, 
            live_data['bearing_change'] / live_data['segment_distance'], 
            0.0)
        live_data['bearing_change_per_second'] = np.where(
            live_data['segment_duration'] > 0, 
            live_data['bearing_change'] / live_data['segment_duration'], 
            0.0)

        # Calculate speed in km/h
        live_data['speed'] = np.where(
            live_data['segment_duration'] > 0,
            live_data['segment_distance'] / live_data['segment_duration'],
            0.0
        )
        # Movement patterns based on calculated metrics
        live_data['acceleration'] = np.where(
            live_data['segment_duration'] > 0, 
            live_data.groupby('trip_id')['speed'].diff(-1) / live_data['segment_duration'], 
            0.0)
        live_data['direction_consistency'] = abs(live_data['bearing_change']).rolling(5).mean()
        
        return live_data
    
    def _compute_route_characteristics(self, live_data: pd.DataFrame, route_id: str):
        """Compute and cache route characteristics"""
        
        route_trips = live_data[live_data['route_id'] == route_id]
        
        characteristics = {
            'avg_trip_length': route_trips['segment_distance'].sum() / route_trips['trip_id'].nunique(),
            'complexity_score': self._calculate_complexity_score(route_trips),
            'stop_density': len(route_trips['stop_id'].unique()) / route_trips['segment_distance'].sum(),
            'avg_speed': route_trips['speed'].mean()
        }
        
        self.route_characteristics[route_id] = characteristics
    
    def _calculate_complexity_score(self, route_trips: pd.DataFrame) -> float:
        """
        Calculate route complexity score based on bearing changes per meter and per second
        A more complex route has:
        1. Higher bearing changes per meter (more turns in shorter distance)
        2. Higher bearing changes per second (more frequent turns)
        3. More sharp turns (larger bearing changes)
        """
        if 'bearing_change_per_meter' not in route_trips.columns or 'bearing_change_per_second' not in route_trips.columns:
            return 1.0  # Default complexity if metrics not available
            
        # Get median values for normalization (exclude stationary segments to avoid zero inflation)
        moving_dist_mask = route_trips['segment_distance'] > 0
        moving_time_mask = route_trips['segment_duration'] > 0
        median_change_per_meter = route_trips.loc[moving_dist_mask, 'bearing_change_per_meter'].abs().median()
        median_change_per_second = route_trips.loc[moving_time_mask, 'bearing_change_per_second'].abs().median()
        mean_change_per_meter = route_trips.loc[moving_dist_mask, 'bearing_change_per_meter'].abs().mean()
        mean_change_per_second = route_trips.loc[moving_time_mask, 'bearing_change_per_second'].abs().mean()

        # Calculate components of complexity
        spatial_complexity = mean_change_per_meter / (median_change_per_meter + 1e-6)
        temporal_complexity = mean_change_per_second / (median_change_per_second + 1e-6)
        
        # Count sharp turns (bearing changes > 45 degrees)
        sharp_turns_ratio = (route_trips['bearing_change'].abs() > 45).mean()
        
        # Combine components with weights
        complexity_score = (
            0.4 * spatial_complexity +    # How twisty the route is spatially
            0.4 * temporal_complexity +   # How frequently turns occur
            0.2 * sharp_turns_ratio      # Proportion of sharp turns
        )
        
        # Normalize to 0-5 range for interpretability
        normalized_score = np.clip(complexity_score, 0, 5)
        
        return normalized_score
