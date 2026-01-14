from data_intelligence.infraction_index_engine.ml_solutions.system_analyzer import model, preprocess, query
from datetime import datetime, timedelta
from bson import ObjectId
from data_intelligence.models.Process import Process, ProcessState

# --- Sample processes for training ---
processes = [
    Process(
        _id=ObjectId(), ocorrencia_id=ObjectId(), PID=1,
        priority=1, allocated_memory=512, program_counter=100,
        cpu_usage=0, gpu_usage=0, io_usage=0,
        cpu_time=0, execution_time=0,
        state=ProcessState.NEW,
        timestamp=datetime.utcnow() - timedelta(minutes=10)
    ),
    Process(
        _id=ObjectId(), ocorrencia_id=ObjectId(), PID=2,
        priority=2, allocated_memory=1024, program_counter=200,
        cpu_usage=45, gpu_usage=20, io_usage=10,
        cpu_time=50, execution_time=120,
        state=ProcessState.RUNNING,
        timestamp=datetime.utcnow() - timedelta(minutes=9)
    ),
    Process(
        _id=ObjectId(), ocorrencia_id=ObjectId(), PID=3,
        priority=3, allocated_memory=2048, program_counter=300,
        cpu_usage=0, gpu_usage=0, io_usage=5,
        cpu_time=30, execution_time=60,
        state=ProcessState.WAITING,
        timestamp=datetime.utcnow() - timedelta(minutes=8)
    ),
    Process(
        _id=ObjectId(), ocorrencia_id=ObjectId(), PID=4,
        priority=4, allocated_memory=4096, program_counter=400,
        cpu_usage=0, gpu_usage=0, io_usage=0,
        cpu_time=200, execution_time=200,
        state=ProcessState.TERMINATED,
        timestamp=datetime.utcnow() - timedelta(minutes=7)
    ),
    Process(
        _id=ObjectId(), ocorrencia_id=ObjectId(), PID=5,
        priority=5, allocated_memory=8192, program_counter=500,
        cpu_usage=70, gpu_usage=40, io_usage=25,
        cpu_time=500, execution_time=800,
        state=ProcessState.RUNNING,
        timestamp=datetime.utcnow() - timedelta(minutes=6)
    ),
]

# --- Process used for prediction ---
prediction_process = Process(
    _id=ObjectId(), ocorrencia_id=ObjectId(), PID=99,
    priority=3, allocated_memory=4096, program_counter=250,
    cpu_usage=55, gpu_usage=30, io_usage=15,
    cpu_time=120, execution_time=200,
    state=ProcessState.RUNNING,
    timestamp=datetime.utcnow() - timedelta(minutes=5)
)

# Path where the model checkpoint will be saved
checkpoint_path = "49949230861"

# Create or load the model
model_bundle = model.create_model(checkpoint_path)

# Preprocess training data (convert Process objects into feature vectors)
train_data = [preprocess(p) for p in processes]

# Train the model with the sample processes
model.train(train_data, model_bundle, checkpoint_path, incremental=False)

# Make a prediction for the test process
try:
    prediction = query.predict(checkpoint_path, prediction_process, incremental=False)
    print("Prediction for test process:", prediction)
except ValueError as e:
    print("Validation error:", e)

# Example of incremental training with a new process
new_process = Process(
    _id=ObjectId(), ocorrencia_id=ObjectId(), PID=100,
    priority=2, allocated_memory=2048, program_counter=150,
    cpu_usage=35, gpu_usage=10, io_usage=5,
    cpu_time=60, execution_time=100,
    state=ProcessState.RUNNING,
    timestamp=datetime.utcnow() - timedelta(minutes=3)
)

try:
    incremental_prediction = query.predict(checkpoint_path, new_process, incremental=True)
    print("Incremental prediction for new process:", incremental_prediction)
except ValueError as e:
    print("Incremental validation error:", e)
