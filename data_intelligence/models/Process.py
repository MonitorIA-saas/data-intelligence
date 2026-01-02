from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from bson import ObjectId


class ProcessState(Enum):
    NEW = 0
    READY = 1
    RUNNING = 2
    WAITING = 3
    TERMINATED = 4


@dataclass
class Process:
    _id: ObjectId
    ocorrencia_id: ObjectId
    PID: int
    priority: int
    allocated_memory: int
    state: ProcessState
    cpu_usage: float
    gpu_usage: float
    io_usage: float
    cpu_time: float = 0.0
    program_counter: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "_id": self._id,
            "ocorrencia_id": self.ocorrencia_id,
            "PID": self.PID,
            "priority": self.priority,
            "allocated_memory": self.allocated_memory,
            "state": self.state.value,
            "cpu_usage": self.cpu_usage,
            "gpu_usage": self.gpu_usage,
            "io_usage": self.io_usage,
            "cpu_time": self.cpu_time,
            "program_counter": self.program_counter,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
        }
