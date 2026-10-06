# AnomalyVFM+

## Extended Zero-Shot Visual Anomaly Detection using Vision Foundation Models

This project is an experimental implementation and extension of the
AnomalyVFM research framework presented at CVPR 2026.

The project aims to investigate whether Vision Foundation Models can be
adapted for zero-shot visual anomaly detection and whether additional
techniques can improve anomaly localization, robustness, and practical
usability.

---

## Project Goal

Our system aims to:

1. Study the AnomalyVFM framework.
2. Reproduce a working baseline.
3. Build a modular implementation.
4. Experiment with Vision Foundation Models.
5. Use parameter-efficient adaptation such as LoRA.
6. Generate and utilize synthetic anomalies.
7. Detect anomalies in unseen categories.
8. Improve anomaly localization and confidence estimation.
9. Evaluate our approach against the baseline.

---

## Planned Pipeline

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
Visualization and Evaluation

---

## Project Structure

- `models/` - Vision models, adapters, decoder and feature fusion
- `synthetic/` - Synthetic anomaly generation and filtering
- `training/` - Training pipeline and loss functions
- `evaluation/` - Metrics and experiment comparison
- `inference/` - Prediction and visualization
- `configs/` - Experiment configurations
- `experiments/` - Experiment records
- `demo/` - Final demonstration application
- `results/` - Generated results, plots and predictions
- `docs/` - Project documentation

---

## Current Status

### Milestone 1 - Environment Setup
- [x] GitHub repository created
- [x] VS Code project setup
- [x] Python virtual environment
- [x] PyTorch installation
- [x] CPU environment verified

### Upcoming

- [ ] Official AnomalyVFM baseline
- [ ] Dataset setup
- [ ] Foundation model integration
- [ ] LoRA adaptation
- [ ] Decoder implementation
- [ ] Synthetic anomaly pipeline
- [ ] Proposed improvements
- [ ] Experiments
- [ ] Evaluation
- [ ] Final demo

---

## Reproducibility

Every major experiment will record:

- Configuration
- Dataset
- Model
- Training parameters
- Random seed
- Metrics
- Checkpoint
- Result visualizations

This allows different team members to reproduce and continue experiments.

---

## Team Development

The project follows a modular architecture so that different team
members can independently work on different components.

Each module will define clear:

`INPUT → PROCESS → OUTPUT`

interfaces.

---

## Research Reference

AnomalyVFM:
"Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors"

CVPR 2026.

The official AnomalyVFM implementation is used as a research reference
and baseline for comparison.
