from data_intelligence.models.Process import Process, ProcessState
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.model import create_model, train
import os
from bson import ObjectId
from datetime import datetime

def verify(process: Process) -> bool:
    # Type checks
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

    # Value ranges
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

    # State logic
    if process.state == ProcessState.TERMINATED:
        if not all([process.cpu_usage == 0, process.gpu_usage == 0, process.io_usage == 0]):
            return False

    elif process.state == ProcessState.NEW:
        if not all([process.cpu_time == 0, process.execution_time == 0]):
            return False

    elif process.state == ProcessState.RUNNING:
        if not any([process.cpu_usage > 0, process.gpu_usage > 0, process.io_usage > 0]):
            return False

    # Temporal coherence
    if process.timestamp > datetime.utcnow():
        return False
    if process.execution_time < process.cpu_time:
        return False

    return True

def predict(checkpoint_path, process: Process):

    if not verify(process):
        raise ValueError('Inconsistent object.')

    if not os.path.exists(checkpoint_path):
        model_bundle = create_model()
    else:
        model_bundle = create_model(checkpoint_path)
    
    train(model_bundle)
    model_bundle[0].predict([process])
