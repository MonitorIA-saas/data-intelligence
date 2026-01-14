from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.IncrementalIsolationForest import IncrementalIsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

def create_model(checkpoint_path):
    global CHECKPOINT_PATH

    if os.path.exists(checkpoint_path):
        return joblib.load(checkpoint_path)

    model_bundle = {
        "model": IncrementalIsolationForest(
            n_estimators=240,
            max_samples=256,
            replace_rate=0.1,
            random_state=1
        ),
        "scaler": StandardScaler()
    }

    joblib.dump(model_bundle, checkpoint_path)
    return model_bundle

def train(train_data, model_bundle, checkpoint_path, incremental=False):
    X = np.array(train_data)

    if not incremental:
        X_scaled = model_bundle["scaler"].fit_transform(X)
        model_bundle["model"].fit(X_scaled)
    else:
        X_scaled = model_bundle["scaler"].transform(X)
        model_bundle["model"].partial_fit(X_scaled)

    joblib.dump(model_bundle, checkpoint_path)
