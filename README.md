# Dddit AI

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Quixel_Texel_Logo" height="400"></p>


## 👋 Author

**Angelo Antonio Prisco** - [AngeloAntonioPrisco](https://github.com/AngeloAntonioPrisco)  

At the moment, I am the only contributor to this project.  
I am a student at **University of Salerno (UNISA)**, currently enrolled in the Master's program in **Software Engineering**.

## 📌 Project Overview

**Dddit AI** is a Python-based system that automatically tags **FBX 3D models** pushed to the **Dddit Server**.  
Its goal is to help 3D artists organize and understand assets, improving workflow efficiency.

**Key Use Cases:**  
- Automatic tagging of 3D models (`character`, `environment`, `prop`, `weapon`, etc.)  
- Track changes between model versions  
- Maintain consistency in large repositories of assets  

**Target Users:**  
- 3D artists specializing in architecture, props, environment, characters, weapons, etc.

## 🧩 Features

- Integration with Dddit Server’s push workflow  
- Assigns tags such as model complexity (`lowpoly`, `highpoly`), type (`prop`, `character`, `environment`), and style (`realistic-style`, `stylized`)  
- MLflow tracking for datasets, experiments, and model versions  
- Models exported in **ONNX** format for lightweight inference  
- Conditional deployment based on monitored performance metrics
- Automatic model reload on Dddit Server. 

## 🚀 How to try it

The entire project can be executed locally using **PyCharm**.

### Run locally
1. Clone the repo:
   ```bash
   git clone https://github.com/AngeloAntonioPrisco/ddditai.git
   ```

2. Open the project in Pycharm.

3. Edit the Run/Debug configuration to load environment variables from a custom *.env* file, like:
    ```bash
    SKETCHFAB_TOKEN_1="sketchfab-token-1"
    SKETCHFAB_TOKEN_2="sketchfab-token-2"
    SKETCHFAB_TOKEN_3="sketchfab-token-3"
    SKETCHFAB_TOKEN_4="sketchfab-token-4"

    MLFLOW_ARTIFACT_URI="ml-flow-artifact-uri"
    MLFLOW_TRACKING_URI="ml-flow-tracking-uri

    AZURE_STORAGE_ACCOUNT_NAME="azure-storage-account-name"
    AZURE_STORAGE_KEY="azure-storage-key"
    AZURE_STORAGE_CONNECTION_STRING="azure-storage-connection-string"
    ```
4. Create a folder named `MLflow` in `AppData\Local` folder.

5. Create in `MLflow` folder a folder named `mlruns` and one named `artifacts`.

6. Open `CMD` and type:
    ```bash
    "path-to-project\.venv\Scripts\activate"
    ```

7. In the same `CMD` type:

    ```bash
    mlflow server --backend-store-uri file:///~/AppData/Local/MLflow/mlruns --default-artifact-root file:///~/AppData/Local/MLflow/artifacts --host 127.0.0.1 --port 5000
    ```
7. Start the application from PyCharm.

## 🧱 Built With

- **[Python](https://www.python.org/)** – Core programming language used for data preparation, modeling, and deployment scripts.  
- **[XGBoost](https://xgboost.readthedocs.io/)** – Gradient Boosted Trees algorithm used for multi-label classification.  
- **[ONNX](https://onnx.ai/)** – Open Neural Network Exchange format for cross-platform model interoperability.  
- **[MLflow](https://mlflow.org/)** – Experiment tracking, dataset versioning, and reproducibility.  
- **[Azure Blob Storage](https://azure.microsoft.com/services/storage/blobs/)** – Storage for 3D models and metadata in the deployment environment.  
- **[SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)** – Synthetic oversampling technique for handling class imbalance during training.
- **[SKETCHFAB APIs](https://docs.sketchfab.com/data-api/v3/index.html)** – APIs used for data retrieving process.


## 🔗 Related resources
- [Dddit Server](https://github.com/AngeloAntonioPrisco/ddditserver): The official Java server for interact with the client.
- [Dddit Client](https://github.com/AngeloAntonioPrisco/ddditclient): The official Python client to interact with the Dddit Server APIs, useful for testing and consuming the server's functionalities.
