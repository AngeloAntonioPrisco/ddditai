import os
import mlflow
import argparse
import onnxmltools
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from onnxmltools.convert.common.data_types import FloatTensorType
from azure.storage.blob import BlobServiceClient

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
    "This experiment implements commit 7801a59 of the Modeling document."
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

def training_mlflow_run(run_id: str, artifact_path: str):
    artifact_local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
    csv_files = [f for f in os.listdir(artifact_local_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in artifact folder")
    csv_file_path = os.path.join(artifact_local_path, csv_files[0])
    df = pd.read_csv(csv_file_path)

    # Create run specific folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Model_from_{run_id or 'manual'}_{timestamp}"

    run_folder = artifact_base_folder / run_name
    models_folder = run_folder / "models"

    models_folder.mkdir(parents=True, exist_ok=True)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Retrieve target label for supervised learning
    tags = df['associated_tag'].unique()
    print(f"Tags founded: {tags}\n")

    with mlflow.start_run(run_name=f"Modeling_from_{run_id}") as run:
        for tag in tags:
            print(f"Training tag: {tag}")

            df['target'] = (df['associated_tag'] == tag).astype(int)

            # Selected all features excluded uid, associated_tag and target
            feature_cols = df.columns.drop(['uid', 'associated_tag', 'target'])

            # Renaming of feature in f0, f1, ...
            new_feature_names = [f'f{i}' for i in range(len(feature_cols))]
            rename_dict = dict(zip(feature_cols, new_feature_names))
            df_renamed = df.rename(columns=rename_dict)

            X = df_renamed[new_feature_names]
            y = df_renamed['target']

            print(f"Dataset dimension: {X.shape}")
            print(f"Number of positive examples: {y.sum()}")
            print(f"Number of negative examples: {len(y) - y.sum()}")

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Apply SMOTE if necessary
            if y_train.nunique() > 1 and y_train.sum() < len(y_train) / 2:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"SMOTE applied: new dimensions {X_train.shape}")
            else:
                print("SMOTE not applied")

            # Train XGBoost
            model = XGBClassifier(eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred))

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric(f"accuracy_{tag}", accuracy)
            mlflow.log_metric(f"precision_{tag}", precision)
            mlflow.log_metric(f"recall_{tag}", recall)
            mlflow.log_metric(f"f1_score_{tag}", f1)

            # Export in ONNX
            initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
            onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
            onnx_file_path = os.path.join(models_folder, f"xgb_model_{tag}.onnx")
            onnxmltools.utils.save_model(onnx_model, onnx_file_path)

            mlflow.log_artifact(str(onnx_file_path), artifact_path="models")
            print(f"Saved ONNX model in: {onnx_file_path}\n")

            # Upload on Azure Blob Storage
            if AZURE_CONNECTION_STRING:
                blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
                container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
                container_client.upload_blob(
                    name=f"training/Training_{timestamp}/xgb_model_{tag}.onnx",
                    data=open(onnx_file_path, "rb"),
                    overwrite=True
                )

        print("Training completed.")

if __name__ == "__main__":
    # This main can be used for manual data balancing of a specific mlflow run that produced a csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--artifact_path", required=True)

    args = parser.parse_args()

    training_mlflow_run(args.run_id, args.artifact_path)