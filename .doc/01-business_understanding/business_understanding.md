# Business Understanding – Dddit AI

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Quixel_Texel_Logo" height="400"></p>

## 1. Project Overview
**Project Name:** Dddit AI  
**Parent Project:** Dddit Server – 3D model and material versioning system  

**Objective:**  
Automatically tag FBX 3D models pushed to Dddit, helping artists understand the type, category, and changes between versions of assets. This facilitates resource organization and improves workflow efficiency.

**Target Users:**  
- 3D artists with specializations in architecture, props, environment, characters, weapons, etc.  
- Users work on FBX models (currently only FBX is supported).

## 2. Business Problem
**Problem Statement:**  
The current process of tagging 3D assets manually is time-consuming and prone to inconsistencies. Dddit AI will automate this process to:  
- Make it clear which resource is being edited.  
- Track changes between model versions.  
- Provide a consistent tagging system for all 3D assets.

**Goals:**  
- Assign relevant tags to each pushed FBX model.  
- Ensure tagging is sufficiently accurate and fast.  
- Maintain resource efficiency on a VM with limited memory (1 GB, 700 MB available for the model).

## 3. Success Criteria
**Functional Requirements:**  
- Automatic assignment of tags to pushed FBX models (e.g., `lowpoly`, `prop`, `realistic-style`).  
- Generate a list of tags even if some input data is missing.  
- Process only one model at a time per user.  

**Non-Functional Requirements:**  
- Fast response time: a few seconds per model (≤ 5 seconds recommended).  
- Accuracy: sufficient/discrete tagging quality (F1-score ≥ 0.7 suggested).  
- Resource constraints: model must run within 700 MB RAM.  

**Monitoring Requirements:**  
- Track average response time.  
- Track average accuracy per tag.  
- Monitor CPU and memory usage.  
- Retrain the model if monitoring alerts indicate performance decay.

## 4. Constraints and Assumptions
- Only FBX file format is supported.  
- No user interaction for tagging is required.  
- No explicit security or availability SLAs are defined.  
- Model training will occur locally; MLflow will be used to track datasets, experiments, and metrics.

## 5. Metrics for Evaluation
| Metric                 | Target/Requirement                                |
|------------------------|--------------------------------------------------|
| F1-score               | ≥ 0.7 (sufficient/discrete accuracy)            |
| Response time          | ≤ 5 seconds per model                             |
| CPU & Memory usage     | Fit within VM constraints (≤ 700 MB RAM)        |
| Tag coverage           | Assign at least some tags even if input is partial |

## 6. Notes
- The system must integrate with Dddit’s push workflow.  
- Tags include model complexity (`lowpoly`, `highpoly`) and type (`prop`, `character`, `environment`, `weapon`, etc.).  
- MLflow will track datasets, experiments, and model versions for reproducibility.  
- Deployment is conditional on meeting performance metrics tracked in MLflow.
