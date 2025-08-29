# Modeling – Dddit AI

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Quixel_Texel_Logo" height="400"></p>

## 1. Overview

**Section Description:**  
This section documents the modeling process for Dddit AI.  
It describes the choice of algorithms, modeling techniques,
and the procedures used to create predictive models.
The goal is to transform the prepared dataset into trained machine
learning models that can distinguish between different `associated_tag`
values (e.g., realistic vs stylized).

**Objectives:**  
- Select appropriate modeling techniques for categorical classification.  
- Train one classifier per `associated_tag` (one-vs-rest approach).  
- Ensure robustness against class imbalance through resampling.  
- Export models in ONNX format for deployment.  
- Maintain reproducibility by logging all artifacts in MLflow.

## 2. Modeling Techniques

**Section Description:**  
This section describes the modeling algorithms chosen
and why they were selected over alternatives. 
Each update should be committed (push) to maintain a clear
history of techniques used during model implementation.

**Objectives:**  
- Select appropriate modeling techniques for categorical classification.
- Track reasons, pros and cons of a defined technique.

**Chosen Algorithm:**  
- **XGBoost Classifier** (`XGBClassifier`):
  - Gradient Boosted Trees optimized for structured/tabular data.  
  - Handles mixed feature types well without strict preprocessing requirements.  
  - Provides strong performance for classification tasks with imbalanced data.  

**Why XGBoost?**  
- A relatively easy algorithm to use as a starting point.
- Robust to noisy features and moderately unbalanced datasets.  
- Interpretable feature importance.  
- Compatible with ONNX export for lightweight inference in production (e.g., on Azure VM B1s).

## 3. Assumptions and Constraints

**Section Description:**  
This section defines the boundaries and assumptions under
which the model will be designed, trained, and deployed.  
It clarifies the limitations of the current scope and the context
in which the solution operates, ensuring that decisions are aligned with realistic conditions.  

**Objective:**  
To explicitly state the constraints and assumptions that guide the development process, preventing misunderstandings  
and providing a clear reference for future iterations or extensions of the project. 

**Assumptions:**  
- Input data has been cleaned and features engineered during the **Data Preparation** stage.  
- Each sample is associated with exactly one `associated_tag`.  
- Dataset is stratified to ensure consistent class distribution during training/testing.  

**Constraints:**  
- Deployment target: Azure B1s VM (1 vCPU, 1 GB RAM). Models must be lightweight and efficient.  
- All models must be exportable to ONNX for cross-platform compatibility.  
- Training and evaluation reproducibility is guaranteed via MLflow tracking.

## 4. Model Building

**Section Description:**  
This section outlines how the models are trained and the systematic steps followed.  
Each update should be committed (push) to maintain a clear
history of the building process.

**Objective:**
Define a guide to follow in order to reach a result that can be evaluated in all aspects.

- **Steps:**  
1. For each `associated_tag`, a binary classification dataset is built (`tag` vs `not tag`).  
2. Feature selection: exclude identifiers (`uid`, `associated_tag`, `target`) and use all remaining features.  
3. Features are renamed to `f0, f1, … fn` for consistency.  
4. Dataset is split into training and testing sets (80/20, stratified).  
5. If class imbalance is detected, apply **SMOTE** oversampling on the training set.  
6. Train **XGBoost Classifier** with `eval_metric='logloss'`.  
7. Evaluate performance on the test set.  
8. Convert the trained model to **ONNX format**.  
9. Log artifacts (ONNX model) in MLflow.

## 5. Model Assessment

**Section Description:**  
This section tracks the trained models using appropriate metrics and
results of evaluation processes.  

**Evaluation Metrics:**  
- **Accuracy** (overall correctness).  
- **Classification Report** (precision, recall, F1-score).

## 6. Model Export and Deployment

**Section Description:**  
This section details how models are prepared for deployment.

**Objective:** 
Define a strategy to guarantee the integration of the model into external components.

**Export Format:**  
- Models are exported to **ONNX** using `onnxmltools`.  
- Initial input type defined as `FloatTensorType([None, n_features])`.  

**Deployment Target:**  
- Azure VM B1s (1 vCPU, 1GB RAM).  
- ONNX chosen to ensure lightweight, cross-platform inference.  
- Models logged in MLflow for version control.

## 7. Notes

- Future iterations may test alternative algorithms (e.g., decision trees, LightGBM) for efficiency.  
- Feature importance analysis will guide further feature selection.  
- For production pipelines, metrics such as inference latency, CPU and RAM usage will be monitored and validated via CI/CD tests.  
