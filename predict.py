

import os
import json
import argparse
import pandas as pd
import pyarrow.parquet as pq
from model import HeuristicModel
from test_data_processor import TestDataProcessor
from feature_pipeline import FeaturePipeline

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

class Predictor:
    def __init__(self):
        self.test_data_processor = TestDataProcessor()
        self.feature_pipeline = FeaturePipeline()
        self.model = HeuristicModel()
        self.model.load(MODEL_PATH)

    def load_and_process_data(self, parquet_path: str) -> pd.DataFrame:
        df = pq.read_table(parquet_path).to_pandas() 
        df = self.test_data_processor.clean_trip_data(df)
        route_id = df['route_id'].iloc[0]
        df = self.test_data_processor.process_trips_by_route(df, route_id)
        df = self.feature_pipeline.fit_transform(df, route_id)
        return df

    def predict(self, input_json_path: str, output_json_path: str) -> pd.DataFrame:
        print(os.getcwd())
        print(os.listdir('/'))
        print(os.listdir('/app'))
        print(input_json_path,output_json_path)
        print(os.path.exists('/app/eval_data'), os.path.exists('/eval_data'))
        print(self.test_data_processor.stops_df.head())
        print(self.test_data_processor.route_sequences_df.head())
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")
        with open(input_json_path, 'r') as f:
            input_json = json.load(f)
        predictions = {}
        for idx, parquet_path in input_json.items():
            df = self.load_and_process_data(parquet_path)
            predicted_arrival_times, predicted_durations = self.model.predict(df)
            predictions[idx] = predicted_arrival_times
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
