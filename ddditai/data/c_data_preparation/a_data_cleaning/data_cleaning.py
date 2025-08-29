import os
import pandas as pd
import mlflow
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from ddditai.data.c_data_preparation.b_feature_construction.feature_construction import feature_construction_mlflow_run

# --- CONFIGURATION ---
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "mlflow")

# --- MAIN MLFLOW PIPELINE ---
EXPERIMENT_NAME = "Sketchfab_Experiment"
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ATTENTION: at the moment mlflow run locally, if moved on a VM or external server make sure to change the path
local_appdata = Path(os.environ['LOCALAPPDATA'])
artifact_base_folder = local_appdata / "MLflow" / "artifacts"
artifact_base_folder.mkdir(parents=True, exist_ok=True)

experiment_description = (
    
"This experiment implements commit 7dc8442 of the Data Understanding document and commit 7bddc87 of Data Preparation document."
)
experiment_tags = {
    "mlflow.note.content": experiment_description,
}

if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=f"file:///{artifact_base_folder.resolve().as_posix()}",
        tags=experiment_tags
    )

mlflow.set_experiment(EXPERIMENT_NAME)

def data_cleaning_mlflow_run(run_id: str, artifact_path: str):
    artifact_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    csv_files = [f for f in os.listdir(artifact_local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in artifact folder")
    csv_file_path = os.path.join(artifact_local_path, csv_files[0])
    df = pd.read_csv(csv_file_path)

    # Create run specific folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Data_Cleaning_from_{run_id or 'manual'}_{timestamp}"

    run_folder = artifact_base_folder / run_name
    csv_folder = run_folder / "cleaned_data"

    csv_folder.mkdir(parents=True, exist_ok=True)

    # Cleaning operation
    df = df.drop(columns=["pbr_type"], errors="ignore")
    df["texture_count"] = df["texture_count"].fillna(df["texture_count"].median())
    df = df[df["face_count"] <= 200_000]
    df = df.drop(columns=["user_tags", "user_categories"], errors="ignore")

    cleaned_csv_path = run_folder / "cleaned_data.csv"
    df.to_csv(cleaned_csv_path, index=False)

    with mlflow.start_run(run_name=f"Data_Cleaning_from_{run_id}") as run:
        mlflow.log_artifact(str(cleaned_csv_path), artifact_path="cleaned_data")
        print(f"Data cleaning completed. CSV saved at {cleaned_csv_path}")

        if mlflow.active_run():
            mlflow.end_run()

        feature_construction_mlflow_run(run.info.run_id, "cleaned_data")

    # Upload on Azure Blob Storage
    if AZURE_CONNECTION_STRING:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        container_client.upload_blob(
            name=f"data_cleaning/Data_Cleaning_{timestamp}/cleaned_data.csv",
            data=open(cleaned_csv_path, "rb"),
            overwrite=True
        )

if __name__ == "__main__":
    # This main can be used for manual data cleaning of a specif mlflow run that produced a csv
    data_cleaning_mlflow_run("", "")
