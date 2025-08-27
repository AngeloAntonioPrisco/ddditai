import os
import requests
import csv
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import mlflow
from azure.storage.blob import BlobServiceClient

# --- CONFIGURATION ---
API_BASE = "https://api.sketchfab.com/v3"

SKETCHFAB_TOKEN = os.getenv("SKETCHFAB_TOKEN")

HEADERS = {"Authorization": SKETCHFAB_TOKEN}

TOTAL_MODELS = 500

BATCH_SIZE = 50

MAX_WORKERS = 4

# --- AZURE CONFIG ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "mlflow")

# --- DIRECTORY FOR CSV ---
BASE_DIR = "data_extraction"
os.makedirs(BASE_DIR, exist_ok=True)


# --- FETCH MODEL DATA FUNCTION ---
def fetch_model_data(uid):
    url = f"{API_BASE}/models/{uid}"
    try:
        resp = requests.get(url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"Error for model {uid}: {resp.status_code}")
            return None
        data = resp.json()
        return [
            uid,
            data.get("isAgeRestricted", False),
            data.get("pbrType", ""),
            data.get("textureCount", 0),
            data.get("vertexCount", 0),
            data.get("materialCount", 0),
            data.get("animationCount", 0),
            [t.get("slug", "") for t in data.get("tags", [])],
            [c.get("name", "") for c in data.get("categories", [])],
            data.get("faceCount", 0)
        ]
    except Exception as e:
        print(f"Exception for model {uid}: {e}")
        return None


# --- BATCH RETRIEVING FUNCTION ---
def fetch_model_uids(total_models, batch_size):
    uids = []
    offset = 0
    while len(uids) < total_models:
        params = {
            "license": "cc0",
            "downloadable": "true",
            "limit": batch_size,
            "offset": offset
        }
        resp = requests.get(f"{API_BASE}/models", headers=HEADERS, params=params)
        if resp.status_code != 200:
            print(f"Error for batch request: {resp.status_code}")
            break
        results = resp.json().get("results", [])
        if not results:
            break
        uids.extend([model["uid"] for model in results])
        offset += batch_size
        time.sleep(0.1)
    return uids[:total_models]


# --- FETCH MODELS ---
print("Retrieving CC0 models...")
model_uids = fetch_model_uids(TOTAL_MODELS, BATCH_SIZE)
print(f"Found {len(model_uids)} models.")

# --- MLFLOW RUN ---
with mlflow.start_run(run_name="Sketchfab_CC0_Data") as run:
    run_name = run.info.run_name
    run_id = run.info.run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder for this run
    run_folder = os.path.join(BASE_DIR, f"{run_name}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    mlflow.log_param("total_models", TOTAL_MODELS)
    mlflow.log_param("batch_size", BATCH_SIZE)

    csv_path = os.path.join(run_folder, "sketchfab_cc0.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "uid", "is_age_restricted", "pbr_type", "texture_count",
            "vertex_count", "material_count", "animation_count",
            "user_tags", "user_categories", "face_count"
        ])

        # Multithreading
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_uid = {executor.submit(fetch_model_data, uid): uid for uid in model_uids}
            for future in as_completed(future_to_uid):
                result = future.result()
                if result:
                    writer.writerow(result)
                    mlflow.log_metric("models_collected", 1)
                    print(f"Processed model {result[0]}")

    # --- LOG CSV IN MLFLOW ---
    mlflow.log_artifact(csv_path, artifact_path="sketchfab_cc0.csv")
    print(f"CSV logged in MLflow: {csv_path}")

    # --- UPLOAD SU AZURE BLOB STORAGE ---
    if AZURE_CONNECTION_STRING:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            with open(csv_path, "rb") as data:
                container_client.upload_blob(name=f"{run_name}_{timestamp}/sketchfab_cc0.csv", data=data,
                                             overwrite=True)
            print(f"CSV successfully uploaded to Azure Blob container '{AZURE_CONTAINER_NAME}'")
        except Exception as e:
            print(f"Error uploading CSV to Azure Blob: {e}")
    else:
        print("Azure connection string not set. Skipping Azure upload.")