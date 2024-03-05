import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow

TARGET_COL = "Potability"

NUMERIC_COLS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, help="Path to raw data")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--val_data", type=str, help="Path to validation dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")

    parser.add_argument("--enable_monitoring", type=str, help="Enable logging to ADX")
    parser.add_argument("--table_name", type=str, default="mlmonitoring", help="Table name in ADX for logging")

    args = parser.parse_args()
    return args

def log_training_data(df, table_name):
    # Implement your logging mechanism here
    pass

def main(args):
    '''Read, split, and save datasets'''

    # Reading Data
    data = pd.read_csv(Path(args.raw_data))

    # Split Data
    random_data = np.random.rand(len(data))
    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test]

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    # Save datasets
    train.to_csv(Path(args.train_data))
    val.to_csv(Path(args.val_data))
    test.to_csv(Path(args.test_data))

    if args.enable_monitoring.lower() in ['true', '1', 'yes']:
        log_training_data(data, args.table_name)

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()

    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Validation dataset output path: {args.val_data}")
    print(f"Test dataset path: {args.test_data}")

    main(args)
    mlflow.end_run()
