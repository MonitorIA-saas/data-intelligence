from data_intelligence.models.Process import Process
from data_intelligence.models.Process import Process, ProcessState
from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer.model import create_model
from bson import ObjectId
from datetime import datetime
import numpy as np

state_mapper = []
with open('data_intelligence\infraction_index_engine\ml_solutions\system_analyzer\state_mapper.txt' , 'r') as file:
    for line in file:
        state_mapper.append(line)


def processes_to_features(processes: list) -> np.ndarray:
    return np.array([
        np.array([
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
    for process in processes if type(process) == Process])


def preprocess(processes) -> list:
    global state_mapper

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

    X = []
    for process in processes:
        if not verify(process):
            raise ValueError("Inconsistent Object")
    
        state_value = process.state
        if state_value not in state_mapper:
            state_mapper.append(state_value)

            with open('data_intelligence\infraction_index_engine\ml_solutions\system_analyzer\state_mapper.txt' , 'a') as file:
                file.write(str(state_value) + '\n')

        state_index = state_mapper.index(state_value)
        timestamp_int = int(process.timestamp.timestamp())

        X.append([
            process.priority,
            process.allocated_memory,
            state_index,
            process.cpu_usage,
            process.gpu_usage,
            process.io_usage,
            process.cpu_time,
            process.program_counter,
            process.execution_time,
            timestamp_int
        ])
    
    return np.array(X)