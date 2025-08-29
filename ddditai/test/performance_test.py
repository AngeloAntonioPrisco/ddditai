import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from azure.storage.blob import BlobServiceClient

# --- CONFIGURATION ---
BLOB_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

MODELS_CONTAINER = "mlflow"

MODELS_PREFIX = "training/"

LOCAL_MODELS_DIR = "models/"

os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

def download_latest_models():
    blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service.get_container_client(MODELS_CONTAINER)

    all_blobs = [b.name for b in container_client.list_blobs(name_starts_with=MODELS_PREFIX)]

    training_folders = set()
    for blob_name in all_blobs:
        parts = blob_name.split("/")
        if len(parts) > 1 and parts[1].startswith("Training_"):
            training_folders.add(parts[1])

    if not training_folders:
        raise FileNotFoundError("Training folder not found.")

    latest_folder = sorted(training_folders, reverse=True)[0]
    print(f"Latest training folder found: {latest_folder}")

    latest_blobs = [b for b in all_blobs if b.startswith(f"{MODELS_PREFIX}{latest_folder}") and b.endswith(".onnx")]

    if not latest_blobs:
        raise FileNotFoundError(f"No ONNX models found in {latest_folder}.")

    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    downloaded_models = []

    for blob_name in latest_blobs:
        local_path = os.path.join(LOCAL_MODELS_DIR, os.path.basename(blob_name))
        with open(local_path, "wb") as f:
            f.write(container_client.download_blob(blob_name).readall())
        downloaded_models.append(local_path)
        print(f"Downloaded model: {local_path}")

    return downloaded_models

def measure_model_performance(model_path, batch_size=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_shape = [batch_size if s is None else s for s in input_shape]

    X_dummy = np.random.rand(*input_shape).astype(np.float32)

    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 ** 2  # MB
    start_time = time.time()

    session.run(None, {input_name: X_dummy})

    elapsed_time = time.time() - start_time
    end_mem = process.memory_info().rss / 1024 ** 2
    mem_used = end_mem - start_mem

    print(f"Model {os.path.basename(model_path)}: time={elapsed_time:.4f}s, RAM usage={mem_used:.2f} MB")
    return elapsed_time, mem_used

def measure_all_models_performance(model_paths, batch_size=1):
    total_time = 0.0
    max_ram = 0.0

    for model_path in model_paths:
        elapsed_time, mem_used = measure_model_performance(model_path, batch_size=batch_size)
        total_time += elapsed_time
        if mem_used > max_ram:
            max_ram = mem_used

    print(f"\nTotal performance for {len(model_paths)} models: total time={total_time:.4f}s, max RAM usage={max_ram:.2f} MB")
    return total_time, max_ram

def main():
    models = download_latest_models()
    measure_all_models_performance(models, batch_size=1)

if __name__ == "__main__":
    main()
