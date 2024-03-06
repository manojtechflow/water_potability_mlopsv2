import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from src.logger import logging

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
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of base estimators")
    parser.add_argument("--max_samples", type=int, default=1.0, help="Number of samples to draw from X to train each base estimator")
    parser.add_argument("--max_features", type=int, default=1.0, help="Number of features to draw from X to train each base estimator")
    args = parser.parse_args()
    return args

def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_csv(Path(args.train_data))

    # Split the data into input(X) and output(y)
    X_train = train_data[NUMERIC_COLS]
    y_train = train_data[TARGET_COL]

    # Train a BaggingClassifier Model with the training set
    model = BaggingClassifier(n_estimators=args.n_estimators,
                              max_samples=args.max_samples,
                              max_features=args.max_features,
                              random_state=42)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Classifier Model
    y_pred_train = model.predict(X_train)

    # Evaluate Classifier performance with the train set
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    
    # Log model hyperparameters and performance metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_samples", args.max_samples)
    mlflow.log_param("max_features", args.max_features)
    mlflow.log_metric("train_accuracy", accuracy_train)
    mlflow.log_metric("train_precision", precision_train)
    mlflow.log_metric("train_recall", recall_train)
    mlflow.log_metric("train_f1", f1_train)

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

    logging.info("Model trained and saved successfully.")

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
