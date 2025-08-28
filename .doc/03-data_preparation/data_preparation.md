# Data Preparation – Dddit AI

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Quixel_Texel_Logo" height="400"></p>

## 1. Overview

**Section Description:**  
This section provides a high-level overview of the entire data preparation process for Dddit AI.  
It describes the sequential steps taken to clean, transform, and prepare the dataset for model training, ensuring robustness and consistency.  
Each update should be committed (push) to maintain a clear history of findings and results obtained in relation to the chosen methodologies.

**Objectives:**  
- Organize raw Sketchfab metadata into a clean, usable dataset.  
- Handle missing values, extreme values, and uninformative features.  
- Construct new features to improve model discriminative power.  
- Ensure the dataset is ready for downstream modeling.  
- Log all results, transformations, and dataset versions using MLflow to maintain traceability and reproducibility.

## 2. Data Cleaning

**Section Description:**  
This section handles missing values, extreme values, and features that are not usable at the model push stage.  
Cleaning ensures that the dataset is consistent and avoids introducing noise into the model.  
Each update should be committed (push) to maintain a clear history of findings and results obtained.

**Objectives:**  
- Remove features with excessive missing data (`pbr type` ~77%).  
- Impute missing numerical values (`texture count` ~43%) using appropriate statistical methods.  
- Remove extreme outliers (`face count` > 200,000) to avoid skewing model training.  
- Drop features not intended for use during model push (`user tags`, `user categories`).  

**Strategies:**  
- **Missing values:** Remove or impute as described above.  
- **Outliers:** Remove extreme models likely from photogrammetry pipelines.  
- **Feature removal:** Drop irrelevant or unavailable features.

## 3. Feature Construction

**Section Description:**  
This phase creates new features to capture information not directly available in the original dataset.  
Each update should be committed (push) to maintain a clear history of findings and results obtained.

**Objectives:**  
- Construct features that improve the model’s ability to discriminate between styles.  
- Reduce dependency on inconsistent or missing features, such as `pbr type`.  

**Strategies:**  
- Introduce `texture richness` to measure visual/textural complexity.  
- This feature helps differentiate realistic vs stylized models.  
- `pbr type` may be unreliable because 3D software (e.g., Maya) can export `Phong`, `Lambert`, or `Non-Specular` materials, which do not indicate whether a model is realistic or stylized.

## 4. Feature Scaling

**Section Description:**  
This phase ensures numerical features are normalized to be comparable across scales and suitable for model input.

**Objectives:**  
- Standardize numerical features to improve model convergence and performance.  
- Facilitate comparisons and distance-based calculations in machine learning algorithms.

**Strategies:**  
- Min-Max normalization  
- Z-score normalization  

## 5. Feature Selection

**Section Description:**  
This phase identifies and removes irrelevant or uninformative features to simplify the model and improve generalization.  
Each update should be committed (push) to maintain a clear history of findings and results obtained.

**Objectives:**  
- Remove features with low variance or little predictive power.  
- Retain features with strong correlation to `associated tag`.  

**Strategies:**  
- **Low variance removal:** Eliminate features with little variability.  
- **Univariate feature selection:** Keep features most predictive of `associated tag`.  

## 6. Data Balancing

**Section Description:**  
This phase ensures balanced class distributions for reliable model training.  
Each update should be committed (push) to maintain a clear history of findings and results obtained.

**Objectives:**  
- Guarantee equal representation for each class to avoid bias.  

**Strategies:**  
- The dataset is generated with the same number of instances per class, ensuring natural balance.  

## 6. Notes
- Scaling can be omitted for initial modeling if the chosen algorithm handles unscaled features.
- Some features, such as `is age restricted`, are likely to be removed because of their low variance and limited predictive power
- Balancing can be skipped for initial modeling since the dataset is already balanced.