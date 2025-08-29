import os
import pandas as pd
from azure.storage.blob import BlobServiceClient

# --- CONFIGURATION ---
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

CONTAINER_NAME = "mlflow"

TRAINING_PREFIX = "training/"

LOCAL_MODELS_DIR = "models/"

os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)


def fetch_latest_models_and_results():
    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)

    all_blobs = [b.name for b in container_client.list_blobs(name_starts_with=TRAINING_PREFIX)]

    training_folders = set()
    for blob_name in all_blobs:
        parts = blob_name.split("/")
        if len(parts) > 1 and parts[1].startswith("Training_"):
            training_folders.add(parts[1])

    if not training_folders:
        raise FileNotFoundError("Training folder not found.")

    latest_folder = sorted(training_folders, reverse=True)[0]
    print(f"Latest training folder found: {latest_folder}")

    model_blobs = [b for b in all_blobs if b.startswith(f"{TRAINING_PREFIX}{latest_folder}/models/") and b.endswith(".onnx")]
    if not model_blobs:
        raise FileNotFoundError(f"No ONNX models found in {latest_folder}.")

    local_models = []
    for blob_name in model_blobs:
        local_path = os.path.join(LOCAL_MODELS_DIR, os.path.basename(blob_name))
        with open(local_path, "wb") as f:
            f.write(container_client.download_blob(blob_name).readall())
        local_models.append(local_path)
        print(f"Downloaded model: {local_path}")

    result_blobs = [b for b in all_blobs if b.startswith(f"{TRAINING_PREFIX}{latest_folder}/results/") and b.endswith(".csv")]
    if not result_blobs:
        raise FileNotFoundError(f"No CSV found in {latest_folder}")

    model_results = {}
    for blob_name in result_blobs:
        result_name = os.path.basename(blob_name).replace("results_", "").replace(".csv", "")
        local_csv_path = os.path.join("results", os.path.basename(blob_name))
        os.makedirs("results", exist_ok=True)
        with open(local_csv_path, "wb") as f:
            f.write(container_client.download_blob(blob_name).readall())
        df = pd.read_csv(local_csv_path)
        model_results[result_name] = df.iloc[0].to_dict()
        print(f"Downloaded results: {local_csv_path}")

    model_result_pairs = []
    for model_path in local_models:
        model_name = os.path.basename(model_path).replace("xgb_model_", "").replace(".onnx", "")
        if model_name in model_results:
            model_result_pairs.append((model_path, model_results[model_name]))
        else:
            print(f"CSV not found for {model_name}")

    print(model_result_pairs)
    return model_result_pairs


if __name__ == "__main__":
    pairs = fetch_latest_models_and_results()
    for model, results in pairs:
        print(f"{os.path.basename(model)} -> {results}")
