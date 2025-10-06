

import os
import json
import argparse
import pandas as pd
import pyarrow.parquet as pq
from model import HeuristicModel
# from ml_model import MLBusArrivalModel
from test_data_processor import TestDataProcessor
from feature_pipeline import FeaturePipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
ML_MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml_model.pkl")

class Predictor:
    def __init__(self):
        self.test_data_processor = TestDataProcessor()
        self.feature_pipeline = FeaturePipeline()
        
        # Try to load ML model first, fallback to heuristic
        self.use_ml_model = False
        self.ml_model = None
        self.heuristic_model = HeuristicModel()
        self.heuristic_model.load(MODEL_PATH)

    def load_and_process_data(self, parquet_path: str) -> pd.DataFrame:
        df = pq.read_table(parquet_path).to_pandas() 
        df = self.test_data_processor.clean_trip_data(df)
        current_data, route_id = self.test_data_processor.get_current_data(df)
        current_df = self.test_data_processor.process_data(current_data, route_id)
        current_df = self.feature_pipeline.fit_transform(current_df, route_id)
        # print(f"DF: {current_df.head()}")
        return current_df, route_id

    def predict(self, input_json_path: str, output_json_path: str) -> pd.DataFrame:
        if not os.path.exists(input_json_path):
            input_json_path = '/app/data/input.json'
        #     raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")
        with open(input_json_path, 'r') as f:
            input_json = json.load(f)
        predictions = {}
        print(input_json)
        for idx, parquet_path in input_json.items():
            print(parquet_path,os.path.exists(parquet_path))
            if not os.path.exists(parquet_path):
                parquet_path = os.path.join('/app/data', parquet_path)
            df, route_id = self.load_and_process_data(parquet_path)
            predicted_arrival_times, predicted_durations = self.heuristic_model.predict(df)
            predictions[route_id] = predicted_arrival_times
            print(f"Predicted arrival times for route {route_id}: {predicted_arrival_times}")
        with open(output_json_path, 'w') as f:
            json.dump(predictions, f)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input-json", type=str, required=True)
    arg_parser.add_argument("--output-json", type=str, required=True)
    args = arg_parser.parse_args()
    input_json_path = args.input_json
    output_json_path = args.output_json
    predictor = Predictor()
    predictor.predict(input_json_path, output_json_path)

if __name__ == "__main__":
    main()
