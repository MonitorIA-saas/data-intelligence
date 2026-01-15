from data_intelligence.models.Process import Process, ProcessState
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.model import create_model
import os
import joblib
from bson import ObjectId
from datetime import datetime
import numpy as np

def verify(process: Process) -> bool:
    if not all([
        isinstance(process._id, ObjectId),
        isinstance(process.ocorrencia_id, ObjectId),
        isinstance(process.PID, int),
        isinstance(process.priority, int),
        isinstance(process.allocated_memory, int),
        isinstance(process.program_counter, int),
        isinstance(process.cpu_usage, (int, float)),
        isinstance(process.gpu_usage, (int, float)),
        isinstance(process.io_usage, (int, float)),
        isinstance(process.cpu_time, (int, float)),
        isinstance(process.execution_time, (int, float)),
        isinstance(process.state, ProcessState),
        isinstance(process.timestamp, datetime),
    ]):
        return False

    if not all([
        process.PID > 0,
        process.allocated_memory >= 0,
        process.program_counter >= 0,
        0 <= process.cpu_usage <= 100,
        0 <= process.gpu_usage <= 100,
        0 <= process.io_usage <= 100,
        process.cpu_time >= 0,
        process.execution_time >= 0,
    ]):
        return False

    if process.state == ProcessState.TERMINATED:
        if not all([process.cpu_usage == 0, process.gpu_usage == 0, process.io_usage == 0]):
            return False

    elif process.state == ProcessState.NEW:
        if not all([process.cpu_time == 0, process.execution_time == 0]):
            return False

    elif process.state == ProcessState.RUNNING:
        if not any([process.cpu_usage > 0, process.gpu_usage > 0, process.io_usage > 0]):
            return False

    if process.timestamp > datetime.utcnow():
        return False
    if process.execution_time < process.cpu_time:
        return False

    return True


def process_to_features(process: Process) -> np.ndarray:
    return np.array([
        process.PID,
        process.priority,
        process.allocated_memory,
        process.program_counter,
        process.cpu_usage,
        process.gpu_usage,
        process.io_usage,
        process.cpu_time,
        process.execution_time,
        process.state.value,
    ], dtype=float)


def load_or_create_model(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        return joblib.load(checkpoint_path)
    return create_model(checkpoint_path)


def save_model(model_bundle, checkpoint_path: str):
    joblib.dump(model_bundle, checkpoint_path)


def predict(checkpoint_path: str, process: Process, incremental: bool = False):
    if not verify(process):
        raise ValueError("Inconsistent object.")

    model_bundle = load_or_create_model(checkpoint_path)

    X = np.array([process_to_features(process)])

    print('THRESHOLD: ' + str(model_bundle['threshold']))

    if not incremental:
        X_scaled = model_bundle["scaler"].transform(X)
        print(model_bundle["model"].decision_function(X_scaled)[0])
        return -1 if model_bundle["model"].decision_function(X_scaled)[0] < model_bundle['threshold'] else 1
    
    X_scaled = model_bundle["scaler"].transform(X)
    model_bundle["model"].partial_fit(X_scaled)
    save_model(model_bundle, checkpoint_path)
    print(model_bundle["model"].decision_function(X_scaled)[0])
    
    return -1 if model_bundle["model"].decision_function(X_scaled)[0] < model_bundle['threshold'] else 1
