from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def create_model(checkpoint_path = None):
    if not checkpoint_path:
        model_info = [
            IsolationForest(
            n_estimators=240,
            max_features="auto",
            bootstrap=False,
            n_jobs=1,
            random_state=1,
            verbose=1,
            warm_start=True
        ),
        StandardScaler()
        ]
        joblib.dump(model_info)
    
    return joblib.load(checkpoint_path)

def train(train_data, model_info):
    X = np.array(train_data)
    X_scaled = model_info[1].fit_transform(X)
    model_info.fit(X_scaled)
    joblib.dump(model_info)
