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
mlflow.set_experiment("Latihan Fraud Detection RandomForest")

# Load preprocessed dataset
data = pd.read_csv("Eksperimen_Fadillah-akbar/preprocessing/data_clean.csv")

# Feature engineering 
# Compute frequency features only if they are not already present.
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

# Training + MLflow Tracking
with mlflow.start_run():
    mlflow.autolog()

    n_estimators = 100
    random_state = 42
    max_depth = 23

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

