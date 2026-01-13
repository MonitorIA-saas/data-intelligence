from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

CHECKPOINT_PATH = "model_checkpoint.joblib"

def create_model(checkpoint_path=CHECKPOINT_PATH):
    global CHECKPOINT_PATH

    if os.path.exists(checkpoint_path) and checkpoint_path != CHECKPOINT_PATH:
        return joblib.load(checkpoint_path)
    
    model_bundle = {
            'model': IsolationForest(
                    n_estimators=240,
                    max_features="auto",
                    bootstrap=False,
                    n_jobs=1,
                    random_state=1,
                    verbose=1,
                    warm_start=True
                ),
            'scaler': StandardScaler()
        }

    joblib.dump(model_bundle, checkpoint_path)
    return model_bundle

def train(train_data, model_bundle, checkpoint_path=CHECKPOINT_PATH):
    X = np.array(train_data)
    
    X_scaled = model_bundle["scaler"].fit_transform(X)
    model_bundle["model"].fit(X_scaled)
    
    joblib.dump(model_bundle, checkpoint_path)
