# Test Case Specification – Model Performance

<p align="center"><img src='https://i.postimg.cc/SNSGrSv2/dddit-ai-upscaled.png' alt="Dddit AI Logo" height="400"></p>

## 1. Overview

**Section Description:**  
This document defines the test cases for evaluating the performance of Dddit AI models.  
Tests ensure that models meet functional and resource requirements before deployment on target infrastructure.

**Objectives:**  
- Validate that models produce outputs without errors.  
- Measure inference latency and RAM consumption.  
- Ensure compatibility with target deployment environments (e.g., Azure VM B1S).  
- Integrate tests into CI/CD pipelines for automated validation.

## 2. Scope

**Section Description:**  
Performance tests cover all latest tracked ONNX with MLflow.  
Functional correctness is assumed from prior model evaluation;
this specification focuses on system and resource performance.

**Objectives:**  
- Verify that models handle expected input shapes and types.  
- Ensure inference time per sample meets required thresholds.  
- Monitor memory usage to prevent exceeding VM constraints.

## 3. Test Types

**Section Description:**  
All tests are automated using `pytest` and are parametrized for multiple models.

**Test Cases:**  

| Test Name | Description | Expected Result | Thresholds |
|-----------|-------------|----------------|------------|
| `test_model_performance` | Measures inference latency, CPU usage, and RAM usage for a batch of synthetic inputs | Model runs without errors, inference completes within limits | **Latency:** ≤ 0.2 sec/sample<br>**RAM:** ≤ 700 MB<br>**CPU:** ≤ 90% |

**Details:**  
- **Input Generation:** Random synthetic input of shape `[N_SAMPLES, N_FEATURES]`.  
- **Resource Monitoring:** Uses `psutil` to track process memory and CPU.  
- **Inference Measurement:** Time measured per batch and averaged per sample.  
- **Automation:** Test runs for each ONNX model in the `models/` directory.

## 4. Execution Guidelines

**Section Description:**  
Tests are intended to run in a controlled environment, similar to the target deployment VM.

**Steps:**  
1. Ensure all ONNX models are in the `models/` directory.  
2. Execute tests via `pytest`, e.g., `pytest test_model_performance.py`.  
3. Review printed metrics for each model: inference time, CPU, and RAM usage.  
4. Assert statements automatically fail the test if thresholds are exceeded.

## 5. Notes

- Thresholds are set based on Azure B1S VM constraints (1 CPU, ~700 MB available RAM).  
- Synthetic inputs allow quick testing without requiring full datasets.  
- Tests can be integrated into CI/CD pipelines for automatic validation on code or model updates.  
- Future extensions may include functional correctness checks alongside performance metrics.
