import os
from azure.storage.blob import BlobServiceClient

# --- CONFIGURATION ---
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

CONTAINER_NAME = "mlflow"

TRAINING_PREFIX = "training/"

LOCAL_MODELS_DIR = "models/"


os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

DEPLOY_LIST_FILE = "deploy_list.txt"

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

    with open(DEPLOY_LIST_FILE, "w") as f:
        for model_path in local_models:
            f.write(f"{os.path.abspath(model_path)}\n")
    print(f"Created deploy list: {DEPLOY_LIST_FILE}")

    return local_models

if __name__ == "__main__":
    fetch_latest_models_and_results()
