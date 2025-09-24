import pandas as pd
import warnings
import pickle
from datetime import timedelta

warnings.filterwarnings('ignore')

class HeuristicModel:
    """Main prediction engine for bus arrival times"""
    
    def __init__(self):
        self.historical_patterns = {}

    def load(self, model_path: str):
        """Load the model from a file"""
        with open(model_path, 'rb') as f:
            self.historical_patterns = pickle.load(f)

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Predict the arrival time for a given dataframe"""
        route_id = test_data['route_id'].iloc[-1]
        current_hour = test_data['hour'].iloc[-1]
        current_dow = test_data['day_of_week'].iloc[-1]
        cumulative_time = test_data['ts'].iloc[-1]
        current_stop_id = test_data['current_stop_id'].iloc[-1]
        remaining_stops = test_data['future_stop_ids'].iloc[-1]
        distance_to_next_route_stop = test_data['distance_to_next_route_stop'].iloc[-1]
        future_segment_distances = test_data['future_segment_distances'].iloc[-1]
        predicted_arrivals, predicted_durations = {}, {}
        patterns = self.historical_patterns[route_id] if route_id in self.historical_patterns else self.historical_patterns['default']
        travel_times = patterns.get('stop_travel_times', pd.DataFrame())
        for i, stop_id in enumerate(remaining_stops):
            # Use historical patterns
            if i == 0:
                mask = travel_times.loc[
                    (travel_times['current_stop_id'] == current_stop_id) &
                    (travel_times['hour'] == current_hour) &
                    (travel_times['day_of_week'] == current_dow)
                ]
                if mask.empty:
                    mask = travel_times.loc[
                        (travel_times['hour'] == current_hour) &
                        (travel_times['day_of_week'] == current_dow)
                    ]
                mean_speed = mask['speed']['mean'].mean()
                estimated_time = distance_to_next_route_stop / mean_speed
            else:
                prev_stop = remaining_stops[i-1]
                mask = travel_times.loc[
                    (travel_times['current_stop_id'] == prev_stop) &
                    (travel_times['hour'] == current_hour) &
                    (travel_times['day_of_week'] == current_dow)
                ]
                if mask.empty:
                    mask = travel_times.loc[
                        (travel_times['hour'] == current_hour) &
                        (travel_times['day_of_week'] == current_dow)
                    ]
                mean_speed = mask['speed']['mean'].mean()
                segment_distance = future_segment_distances[i-1]
                estimated_time = segment_distance / mean_speed
            predicted_durations[stop_id] = estimated_time
            cumulative_time += timedelta(seconds=estimated_time)
            predicted_arrivals[stop_id] = cumulative_time.strftime('%Y-%m-%d %H:%M:%S')
        return predicted_arrivals, predicted_durations
            