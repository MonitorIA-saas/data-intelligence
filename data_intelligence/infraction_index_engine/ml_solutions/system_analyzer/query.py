from data_intelligence.models.Process import Process, ProcessState
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.model import create_model
import os
import joblib
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.preprocess import preprocess
import numpy as np

def load_or_create_model(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        return joblib.load(checkpoint_path)
    return create_model(checkpoint_path)


def save_model(model_bundle, checkpoint_path: str):
    joblib.dump(model_bundle, checkpoint_path)


def predict(checkpoint_path: str, process: Process, incremental: bool = False):
    model_bundle = load_or_create_model(checkpoint_path)

    X = preprocess(processes=[process])

    if not incremental:
        X_scaled = model_bundle["scaler"].transform(X)
        return [-1 if model_bundle["model"].decision_function(X_scaled)[0] > model_bundle['threshold'] else 1, model_bundle['model'].decision_function(X_scaled)[0]]
    
    X_scaled = model_bundle["scaler"].transform(X)
    model_bundle["model"].partial_fit(X_scaled)
    save_model(model_bundle, checkpoint_path)
    
    return [-1 if model_bundle["model"].decision_function(X_scaled)[0] > model_bundle['threshold'] else 1, model_bundle['model'].decision_function(X_scaled)[0]]

