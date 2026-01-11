from data_intelligence.models.Process import Process
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.model import create_model, train
import os

def predict(checkpoint_path = "", process: Process = None):
    if not os.path.exists(checkpoint_path):
        model_info = create_model()
    else:
        model_info = create_model(checkpoint_path)
    
    train(model_info)
    model_info[0].predict([process])
    