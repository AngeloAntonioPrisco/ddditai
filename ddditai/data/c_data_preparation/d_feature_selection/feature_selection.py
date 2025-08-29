import os
import mlflow
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from ddditai.data.c_data_preparation.e_data_balancing.data_balancing import data_balancing_mlflow_run

# ---- CONFIGURATION ----
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

def feature_selection_mlflow_run(run_id: str, artifact_path: str):
    artifact_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    csv_files = [f for f in os.listdir(artifact_local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in artifact folder")
    csv_file_path = os.path.join(artifact_local_path, csv_files[0])
    df = pd.read_csv(csv_file_path)

    # Create run specific folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Feature_Construction_from_{run_id or 'manual'}_{timestamp}"

    run_folder = artifact_base_folder / run_name
    csv_folder = run_folder / "selected_data"

    csv_folder.mkdir(parents=True, exist_ok=True)

    # Feature Selection
    # Drop low variance / uninformative features
    min_variance_threshold = 1e-3
    max_missing_ratio = 0.9

    columns_to_drop = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var() < min_variance_threshold:
                columns_to_drop.append(col)
                continue

        missing_ratio = df[col].isna().sum() / len(df)
        if missing_ratio > max_missing_ratio:
            columns_to_drop.append(col)

    # Drop identified columns
    if columns_to_drop:
        print(f"Dropping low-variance or uninformative columns: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print("No low-variance or uninformative columns found.")

    selected_csv_path = run_folder / "selected_features.csv"
    df.to_csv(selected_csv_path, index=False)

    with mlflow.start_run(run_name=f"Feature_Selection_from_{run_id}") as run:
        mlflow.log_artifact(str(selected_csv_path), artifact_path="selected_features")

        if mlflow.active_run():
            mlflow.end_run()

        data_balancing_mlflow_run(run.info.run_id, "selected_features")

    # Upload on Azure Blob Storage
    if AZURE_CONNECTION_STRING:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        container_client.upload_blob(
            name=f"feature_selection/Feature_Selection_{timestamp}/selected_features.csv",
            data=open(selected_csv_path, "rb"),
            overwrite=True
        )

    print(f"Feature selection completed. CSV saved at {selected_csv_path}")


if __name__ == "__main__":
    # This main can be used for manual feature selection of a specific mlflow run that produced a csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--artifact_path", required=True)

    args = parser.parse_args()

    feature_selection_mlflow_run(args.run_id, args.artifact_path)
