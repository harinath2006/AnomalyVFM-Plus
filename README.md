# AnomalyVFM+

## Extended Zero-Shot Visual Anomaly Detection using Vision Foundation Models

AnomalyVFM+ is an extended implementation inspired by the CVPR 2026 research work **"AnomalyVFM: Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors."**

The project investigates how a pretrained Vision Foundation Model can be adapted for visual anomaly detection using parameter-efficient learning while extending the pipeline with:

- Multi-scale feature fusion
- LoRA-based parameter-efficient adaptation
- Dedicated image-level anomaly prediction
- Adaptive anomaly scoring
- Confidence-aware anomaly analysis
- Automatic threshold calibration
- Synthetic anomaly generation
- Pixel-level anomaly localization
- Explainable anomaly visualization
- Systematic baseline comparison
- Controlled ablation experiments
- Interactive Gradio inference

The system is designed for industrial visual inspection, where the objective is to determine whether an input image is **normal or anomalous** and identify the spatial location of the anomaly.

> **Important:** AnomalyVFM+ is an extended implementation inspired by the original AnomalyVFM work. It is not claimed to be an exact reproduction of the complete official training pipeline.

---

# Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Main Features](#main-features)
- [Key Contributions](#key-contributions)
- [Model Architecture](#model-architecture)
- [Overall Processing Pipeline](#overall-processing-pipeline)
- [Detailed Methodology](#detailed-methodology)
- [1. Input Processing](#1-input-processing)
- [2. Vision Foundation Model](#2-vision-foundation-model)
- [3. LoRA Adaptation](#3-lora-adaptation)
- [4. Multi-Scale Feature Extraction](#4-multi-scale-feature-extraction)
- [5. Multi-Scale Feature Fusion](#5-multi-scale-feature-fusion)
- [6. Anomaly Decoder](#6-anomaly-decoder)
- [7. Image-Level Anomaly Predictor](#7-image-level-anomaly-predictor)
- [8. Adaptive Anomaly Scoring](#8-adaptive-anomaly-scoring)
- [9. Confidence-Aware Refinement](#9-confidence-aware-refinement)
- [10. Synthetic Anomaly Generation](#10-synthetic-anomaly-generation)
- [Training Strategy](#training-strategy)
- [Loss Functions](#loss-functions)
- [Threshold Calibration](#threshold-calibration)
- [Inference](#inference)
- [Dataset](#dataset)
- [Dataset Structure](#dataset-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Setup](#experimental-setup)
- [Experimental Results](#experimental-results)
- [Baseline Comparison](#baseline-comparison)
- [Ablation Study](#ablation-study)
- [Visualization](#visualization)
- [Interactive Demo](#interactive-demo)
- [Project Structure](#project-structure)
- [Module Responsibilities](#module-responsibilities)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference Usage](#inference-usage)
- [Visualization Usage](#visualization-usage)
- [Demo Usage](#demo-usage)
- [Reproducibility](#reproducibility)
- [Checkpoint Handling](#checkpoint-handling)
- [Technical Design](#technical-design)
- [Experimental Findings](#experimental-findings)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Project Status](#project-status)
- [Conclusion](#conclusion)

---

# Project Overview

Traditional anomaly detection systems often require a large number of labeled anomalous samples.

In industrial inspection, defective samples can be:

- Rare
- Expensive to collect
- Difficult to annotate
- Highly variable
- Different across products and manufacturing environments

AnomalyVFM+ explores a different approach by using a pretrained **Vision Foundation Model** as the visual representation backbone.

The system adapts the pretrained model using **LoRA**, extracts information from multiple transformer layers, fuses those representations, and produces both:

1. An image-level anomaly score.
2. A pixel-level anomaly map.

The overall idea is:

```text
                    Input Image
                         |
                         v
                +----------------+
                |    DINOv2      |
                | Vision Model   |
                +-------+--------+
                        |
                        v
                 LoRA Adaptation
                        |
             +----------+----------+
             |                     |
             v                     v
      Multi-Scale Features      CLS Token
             |                     |
             v                     v
       Feature Fusion       Image Predictor
             |                     |
             v                     v
       Anomaly Decoder       Image Score
             |
             v
       Anomaly Heatmap
             |
             +-----------+
                         |
                         v
                 Threshold Decision
                         |
                 +-------+-------+
                 |               |
                 v               v
              NORMAL         ANOMALOUS