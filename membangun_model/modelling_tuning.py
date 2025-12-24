import mlflow
import pandas as pd
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Fraud Detection RF - Tuning")

# Load dataset
data = pd.read_csv("Eksperimen_Fadillah-akbar/preprocessing/data_clean.csv")

# Feature engineering
if "dest_freq" not in data.columns:
    if "nameDest" in data.columns:
        data["dest_freq"] = data["nameDest"].map(data["nameDest"].value_counts())
    else:
        raise KeyError(
            "Neither 'dest_freq' nor 'nameDest' found in data. Re-run preprocessing or provide the raw columns."
        )

if "orig_freq" not in data.columns:
    if "nameOrig" in data.columns:
        data["orig_freq"] = data["nameOrig"].map(data["nameOrig"].value_counts())
    else:
        raise KeyError(
            "Neither 'orig_freq' nor 'nameOrig' found in data. Re-run preprocessing or provide the raw columns."
        )

# Prepare features and target. Drop identifier columns only if present.
drop_cols = [c for c in ["nameDest", "nameOrig"] if c in data.columns]
X = data.drop(columns=["isFraud"] + drop_cols)
y = data["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

input_example = X_train.iloc[:5]

# Hyperparameter candidates
n_estimators_list = [100, 300]
max_depth_list = [None, 10]

# Manual Tuning + Logging
for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:

        with mlflow.start_run():

            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("model_type", "RandomForestClassifier")

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            # Save model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example
            )
