import os
import time
from pathlib import Path

import requests
import mlflow
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from azure.storage.blob import BlobServiceClient

from ddditai.data.b_data_analysis.data_analysis import analyze_mlflow_run

# --- CONFIGURATION ---
API_BASE = "https://api.sketchfab.com/v3"

TOTAL_MODELS_PER_TAG = 512

BATCH_SIZE = 16

MAX_WORKERS = 4

PAUSE_BETWEEN_REQUESTS = 2  # seconds

PAUSE_EVERY_N_REQUESTS = 32

PAUSE_DURATION = 90  # seconds

SEARCH_TAGS = ["lowpoly", "highpoly", "prop", "character", "environment", "weapon", "realistic", "stylized"]

# ATTENTION: at the moment mlflow run locally, if moved on a VM or external server make sure to change the path
local_appdata = Path(os.environ['LOCALAPPDATA'])
data_folder = local_appdata / "MLflow" / "data_extraction"
data_folder.mkdir(parents=True, exist_ok=True)
BASE_DIR = data_folder

os.makedirs(BASE_DIR, exist_ok=True)

AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "mlflow")

TOKENS = [
    os.getenv("SKETCHFAB_TOKEN_1"),
    os.getenv("SKETCHFAB_TOKEN_2"),
    os.getenv("SKETCHFAB_TOKEN_3"),
    os.getenv("SKETCHFAB_TOKEN_4")
]

# --- REQUEST FUNCTIONS ---
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def request_with_backoff(url, params=None, headers=None, max_retries=10):
    backoff = 1
    retries = 0
    while retries < max_retries:
        try:
            resp = requests.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    time.sleep(int(retry_after) + 1)
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 3600)
                retries += 1
                continue
            time.sleep(backoff)
            backoff = min(backoff * 2, 3600)
            retries += 1
        except requests.exceptions.RequestException:
            time.sleep(backoff)
            backoff = min(backoff * 2, 3600)
            retries += 1
    return requests.Response()

def fetch_model_data(uid, token, thread_name="Thread"):
    headers = {"Authorization": f"Token {token}"}
    url = f"{API_BASE}/models/{uid}"
    time.sleep(PAUSE_BETWEEN_REQUESTS)
    try:
        resp = request_with_backoff(url, headers=headers)
        if resp.status_code != 200:
            return None, None
        data = resp.json()
        tags = [t.get("slug", "") for t in data.get("tags", [])]
        if "noAI" in tags:
            return None, None
        tags = ["realistic-style" if t == "realistic" else t for t in tags]
        model_info = [
            uid,
            "",  # placeholder tag, will be populated after
            data.get("isAgeRestricted", False),
            data.get("pbrType", ""),
            data.get("textureCount", 0),
            data.get("vertexCount", 0),
            data.get("materialCount", 0),
            data.get("animationCount", 0),
            tags,
            [c.get("name", "") for c in data.get("categories", [])],
            data.get("faceCount", 0)
        ]
        author_info = (uid, data.get("user", {}).get("displayName", "unknown"))
        return model_info, author_info
    except Exception:
        return None, None

def fetch_model_uids(tag, total_models, token, thread_name="Thread"):
    uids = []
    offset = 0
    requests_count = 0
    headers = {"Authorization": f"Token {token}"}
    while len(uids) < total_models:
        params = {"tags": tag, "limit": BATCH_SIZE, "offset": offset}
        resp = request_with_backoff(f"{API_BASE}/models", params=params, headers=headers)
        requests_count += 1
        if requests_count % PAUSE_EVERY_N_REQUESTS == 0:
            print(f"[{now()}] [{thread_name}] Pause of {PAUSE_DURATION} sec after {requests_count} requests for tag '{tag}'")
            time.sleep(PAUSE_DURATION)
        if resp.status_code != 200:
            break
        results = resp.json().get("results", [])
        if not results:
            break
        for model in results:
            uid = model["uid"]
            tags_model = [t["slug"] for t in model.get("tags", [])]
            if "noAI" not in tags_model:
                uids.append(uid)
        offset += BATCH_SIZE
        time.sleep(PAUSE_BETWEEN_REQUESTS)
    return uids[:total_models]

def fetch_model_data_with_tag(uid, tag, token, thread_name="Thread"):
    model_info, author_info = fetch_model_data(uid, token, thread_name)
    if model_info:
        tag_value = "realistic-style" if tag == "realistic" else tag
        model_info[1] = tag_value
    return model_info, author_info

def worker_thread(thread_tags, token, thread_name):
    start_time = now()
    print(f"[{start_time}] [{thread_name}] Started with tag: {thread_tags}")
    thread_results = []

    for tag in thread_tags:
        print(f"[{now()}] [{thread_name}] Starting collection of UID for tag '{tag}'")
        model_uids = fetch_model_uids(tag, TOTAL_MODELS_PER_TAG // MAX_WORKERS, token, thread_name=thread_name)
        total_models_tag = len(model_uids)
        print(f"[{now()}] [{thread_name}] Collected {total_models_tag} UID for tag '{tag}'")

        for idx, uid in enumerate(model_uids, 1):
            model_info, author_info = fetch_model_data_with_tag(uid, tag, token, thread_name=thread_name)
            thread_results.append((model_info, author_info))
            print(f"[{now()}] [{thread_name}] Analyzed model {idx}/{total_models_tag} with UID '{uid}' for tag '{tag}'")

    end_time = now()
    print(f"[{end_time}] [{thread_name}] terminated")
    return thread_results

# --- MAIN MLRUN PIPELINE ---
with mlflow.start_run(run_name="Sketchfab_Data") as run:
    run_name = run.info.run_name
    run_id = run.info.run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(BASE_DIR, f"{run_name}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    csv_path = os.path.join(run_folder, "sketchfab_models.csv")
    txt_path = os.path.join(run_folder, "sketchfab_authors.txt")

    mlflow.log_artifact("tags", str(SEARCH_TAGS))
    mlflow.log_param("total_models_per_tag", TOTAL_MODELS_PER_TAG)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("max_workers", MAX_WORKERS)
    mlflow.log_param("batch_size", PAUSE_BETWEEN_REQUESTS)
    mlflow.log_param("max_workers", PAUSE_DURATION)
    mlflow.log_param("batch_size", PAUSE_DURATION)

    # Tags association for thread
    thread_tag_chunks = [SEARCH_TAGS[i::MAX_WORKERS] for i in range(MAX_WORKERS)]

    all_models = []
    all_authors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(txt_path, "w", encoding="utf-8") as txtfile:

        futures = []
        for i in range(MAX_WORKERS):
            futures.append(executor.submit(worker_thread, thread_tag_chunks[i], TOKENS[i], f"Thread-{i+1}"))

        for future in as_completed(futures):
            thread_results = future.result()
            for model_info, author_info in thread_results:
                if model_info:
                    all_models.append(model_info)
                if author_info:
                    txtfile.write(f"{author_info[0]},{author_info[1]}\n")

    # Dataframe creation
    df = pd.DataFrame(all_models, columns=[
        "uid", "associated_tag", "is_age_restricted", "pbr_type", "texture_count",
        "vertex_count", "material_count", "animation_count",
        "user_tags", "user_categories", "face_count"
    ])
    df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path, artifact_path="sketchfab_models.csv")

    # Upload on Azure Blob Storage
    if AZURE_CONNECTION_STRING:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
            container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            with open(csv_path, "rb") as data:
                container_client.upload_blob(
                    name=f"{run_name}_{timestamp}/sketchfab_models.csv", data=data, overwrite=True
                )
            with open(txt_path, "rb") as data:
                container_client.upload_blob(
                    name=f"{run_name}_{timestamp}/sketchfab_authors.txt", data=data, overwrite=True
                )
            print(f"[{now()}] CSV and TXT uploaded to '{AZURE_CONTAINER_NAME}' Azure container")
        except Exception as e:
            print(f"[{now()}] Error during Azure uploading : {e}")

analyze_mlflow_run(run_id, "sketchfab_models.csv")