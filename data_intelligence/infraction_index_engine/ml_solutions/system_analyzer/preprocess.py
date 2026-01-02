from data_intelligence.models.Process import Process

state_mapper = []


def preprocess(process: Process) -> list:
    global state_mapper

    state_value = process.state
    if state_value not in state_mapper:
        state_mapper.append(state_value)
    state_index = state_mapper.index(state_value)

    timestamp_int = int(process.timestamp.timestamp())

    return [
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
    ]