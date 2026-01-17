from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer import model, preprocess, query
from datetime import datetime, timedelta
from bson import ObjectId
from data_intelligence.models.Process import Process, ProcessState
import csv

processes = []

with open("tests\system_analyzer\processes.csv", mode="r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        process = Process(
            _id=ObjectId(),
            ocorrencia_id=ObjectId(),
            PID=int(row["PID"]),
            priority=int(row["priority"]),
            allocated_memory=int(row["allocated_memory"]),
            program_counter=int(row["program_counter"]),
            cpu_usage=int(row["cpu_usage"]),
            gpu_usage=int(row["gpu_usage"]),
            io_usage=int(row["io_usage"]),
            cpu_time=int(row["cpu_time"]),
            execution_time=int(row["execution_time"]),
            state=ProcessState[row["state"]],
            timestamp=datetime.fromisoformat(row["timestamp"])
        )
        processes.append(process)

normal_process = Process(
    _id=ObjectId(),
    ocorrencia_id=ObjectId(),
    PID=100,
    priority=2,
    allocated_memory=512,
    program_counter=150,
    cpu_usage=15,
    gpu_usage=0,
    io_usage=10,
    cpu_time=150,
    execution_time=250,
    state=ProcessState.RUNNING,
    timestamp=datetime.utcnow() - timedelta(seconds=50)
)

anomalous_process = Process(
    _id=ObjectId(),
    ocorrencia_id=ObjectId(),
    PID=201,
    priority=10,
    allocated_memory=16384,
    program_counter=999999,
    cpu_usage=100,
    gpu_usage=100,
    io_usage=100,
    cpu_time=5000,
    execution_time=5000,
    state=ProcessState.RUNNING,
    timestamp=datetime.utcnow() - timedelta(seconds=1)
)

checkpoint_path = "49949230861"

model_bundle = model.create_model(checkpoint_path)

train_data = preprocess(processes)

model.train(train_data, model_bundle, checkpoint_path, incremental=False)

try:
    prediction = query.predict(checkpoint_path, anomalous_process, incremental=False)
    print("Prediction for test process (anomalous):", prediction)

    prediction = query.predict(checkpoint_path, normal_process, incremental=False)
    print("Prediction for test process (normal):", prediction)
except ValueError as e:
    print("Validation error:", e)

try:
    incremental_prediction = query.predict(checkpoint_path, normal_process, incremental=True)
    print("Incremental prediction for new process (normal):", incremental_prediction)

    incremental_prediction = query.predict(checkpoint_path, anomalous_process, incremental=True)
    print("Incremental prediction for new process (anomalous):", incremental_prediction)
except ValueError as e:
    print("Incremental validation error:", e)