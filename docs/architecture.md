# AnomalyVFM+ Architecture

## 1. Project Objective

AnomalyVFM+ is an experimental extension of the AnomalyVFM framework for
zero-shot visual anomaly detection using Vision Foundation Models.

The system is designed to detect whether an image contains an anomaly
and identify the location of the anomalous region.

---

## 2. High-Level Pipeline

Input Image
    ↓
Vision Foundation Model
    ↓
Feature Extraction
    ↓
Parameter-Efficient Adaptation
    ↓
Feature Fusion
    ↓
Anomaly Decoder
    ↓
Anomaly Score + Anomaly Map
    ↓
Visualization + Evaluation

---

## 3. Module Responsibilities

### 3.1 Foundation Model

Location:

`models/foundation/`

Responsibility:

- Load the selected Vision Foundation Model.
- Preprocess input images.
- Extract visual features.
- Provide feature representations to downstream modules.

Input:

`Image Tensor`

Output:

`Feature Maps`

Possible models:

- DINOv2
- RADIO
- Other compatible Vision Foundation Models

---

### 3.2 Adapter

Location:

`models/adapters/`

Responsibility:

- Adapt pretrained foundation-model features for anomaly detection.
- Implement parameter-efficient methods such as LoRA.

Input:

`Foundation Model Features`

Output:

`Adapted Features`

---

### 3.3 Feature Fusion

Location:

`models/fusion/`

Responsibility:

- Combine information from multiple feature levels.
- Support our proposed multi-scale feature fusion experiments.

Input:

`Adapted Feature Maps`

Output:

`Fused Feature Representation`

---

### 3.4 Anomaly Decoder

Location:

`models/decoder/`

Responsibility:

- Convert visual features into anomaly predictions.
- Produce spatial anomaly maps.
- Produce image-level anomaly scores.

Input:

`Feature Representation`

Output:

- `Anomaly Map`
- `Anomaly Score`

---

### 3.5 Synthetic Data

Location:

`synthetic/`

Responsibility:

- Generate synthetic anomalous samples.
- Generate anomaly masks.
- Filter poor-quality synthetic samples.
- Prepare data for training experiments.

Input:

`Normal Images`

Output:

- `Synthetic Anomalous Images`
- `Anomaly Masks`
- `Confidence Information`

---

### 3.6 Training

Location:

`training/`

Responsibility:

- Train adapters and decoder.
- Implement loss functions.
- Manage optimization and learning-rate scheduling.

Input:

`Training Dataset`

Output:

`Model Checkpoint`

---

### 3.7 Evaluation

Location:

`evaluation/`

Responsibility:

- Evaluate trained models.
- Calculate image-level metrics.
- Calculate pixel-level metrics.
- Compare baseline and proposed methods.

Possible metrics:

- Image AUROC
- Pixel AUROC
- F1 score
- Precision
- Recall
- AUPRO

---

### 3.8 Inference

Location:

`inference/`

Responsibility:

- Load trained checkpoints.
- Run predictions on new images.
- Generate anomaly maps.
- Generate visualization overlays.

Input:

`Image`

Output:

- `Anomaly Score`
- `Anomaly Map`
- `Visualization`

---

## 4. Module Contract

Every module should follow:

INPUT → PROCESS → OUTPUT

A module should not directly depend on internal implementation details
of another module.

Example:

Foundation Model:

Image → Feature Maps

Adapter:

Feature Maps → Adapted Features

Decoder:

Adapted Features → Anomaly Map + Score

This allows individual modules to be replaced without rewriting the
entire system.

---

## 5. Baseline

The first implementation will reproduce the core AnomalyVFM pipeline.

Baseline:

Foundation Model
→ LoRA
→ Decoder
→ Anomaly Score + Mask

---

## 6. Proposed Extension

Our proposed experiments will investigate:

1. Multi-scale feature fusion.
2. Adaptive anomaly thresholding.
3. Confidence-aware prediction analysis.
4. Cross-domain evaluation.
5. Synthetic anomaly quality and quantity analysis.

These will only be considered improvements if experimental results
support the claim.

---

## 7. Reproducibility

Every experiment should record:

- Experiment ID
- Dataset
- Model
- Configuration
- Hyperparameters
- Random seed
- Training duration
- Checkpoint
- Evaluation metrics
- Result visualizations

Example:

EXP-001
    ↓
Configuration
    ↓
Model
    ↓
Checkpoint
    ↓
Evaluation
    ↓
Metrics