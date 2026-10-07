import json
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = Path(
    "results/evaluation/final_results.json"
)


# ============================================================
# EXPERIMENTAL RESULTS
# ============================================================

RESULTS = {

    "dataset": {
        "name": "MVTec AD",
        "category": "Bottle",
        "test_images": 83,
        "normal_images": 20,
        "anomalous_images": 63
    },

    "final_model": {
        "name": "AnomalyVFM+",
        "image_auroc": 0.9595,
        "pixel_auroc": 0.8627
    },

    "baseline": {
        "name": "DINOv2 Baseline",
        "image_auroc": 0.4683,
        "pixel_auroc": 0.8857
    },

    "ablations": {

        "single_scale_lora": {
            "name": "Single-scale LoRA",
            "image_auroc": 0.1730,
            "pixel_auroc": 0.5515
        },

        "multi_scale": {
            "name": "Multi-scale Feature Fusion",
            "image_auroc": 0.9595,
            "pixel_auroc": 0.8602,
            "image_improvement": 0.7865,
            "pixel_improvement": 0.3087,
            "image_relative_change_percent": 454.59,
            "pixel_relative_change_percent": 55.98
        },

        "adaptive_scoring": {
            "name": "Adaptive Anomaly Scoring",
            "maximum_score_auroc": 0.9119,
            "adaptive_score_auroc": 0.9444,
            "improvement": 0.0325,
            "relative_change_percent": 3.57
        },

        "confidence_refinement": {
            "name": "Confidence-aware Refinement",
            "without_refinement_pixel_auroc": 0.8602,
            "with_refinement_pixel_auroc": 0.8469,
            "improvement": -0.0134,
            "relative_change_percent": -1.56
        }
    }
}


# ============================================================
# CALCULATE FINAL IMPROVEMENTS
# ============================================================

def calculate_improvements():

    baseline = RESULTS["baseline"]
    final_model = RESULTS["final_model"]

    image_difference = (
        final_model["image_auroc"]
        -
        baseline["image_auroc"]
    )

    pixel_difference = (
        final_model["pixel_auroc"]
        -
        baseline["pixel_auroc"]
    )

    image_relative = (
        image_difference
        /
        baseline["image_auroc"]
        *
        100
    )

    pixel_relative = (
        pixel_difference
        /
        baseline["pixel_auroc"]
        *
        100
    )

    RESULTS["final_vs_baseline"] = {

        "image_auroc_difference":
            round(image_difference, 4),

        "pixel_auroc_difference":
            round(pixel_difference, 4),

        "image_relative_change_percent":
            round(image_relative, 2),

        "pixel_relative_change_percent":
            round(pixel_relative, 2)
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            RESULTS,
            file,
            indent=4
        )

    print()
    print(
        "Final results saved:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary():

    dataset = RESULTS["dataset"]
    baseline = RESULTS["baseline"]
    final_model = RESULTS["final_model"]
    ablations = RESULTS["ablations"]
    improvement = RESULTS["final_vs_baseline"]

    print()
    print("=" * 78)
    print("ANOMALYVFM+ FINAL EXPERIMENTAL RESULTS")
    print("=" * 78)

    print()

    print("DATASET")
    print("-" * 78)

    print(
        f"Dataset           : {dataset['name']}"
    )

    print(
        f"Category          : {dataset['category']}"
    )

    print(
        f"Test images       : {dataset['test_images']}"
    )

    print(
        f"Normal images     : {dataset['normal_images']}"
    )

    print(
        f"Anomalous images  : {dataset['anomalous_images']}"
    )

    print()
    print("MAIN MODEL COMPARISON")
    print("-" * 78)

    print(
        f"{'Model':<30}"
        f"{'Image AUROC':>18}"
        f"{'Pixel AUROC':>18}"
    )

    print("-" * 78)

    print(
        f"{baseline['name']:<30}"
        f"{baseline['image_auroc']:>18.4f}"
        f"{baseline['pixel_auroc']:>18.4f}"
    )

    print(
        f"{final_model['name']:<30}"
        f"{final_model['image_auroc']:>18.4f}"
        f"{final_model['pixel_auroc']:>18.4f}"
    )

    print("-" * 78)

    print(
        f"{'Absolute improvement':<30}"
        f"{improvement['image_auroc_difference']:>18.4f}"
        f"{improvement['pixel_auroc_difference']:>18.4f}"
    )

    print()

    print(
        "Relative Image AUROC change:",
        f"+{improvement['image_relative_change_percent']:.2f}%"
    )

    print(
        "Relative Pixel AUROC change:",
        f"{improvement['pixel_relative_change_percent']:+.2f}%"
    )

    print()
    print("ABLATION STUDY")
    print("-" * 78)

    multi_scale = ablations[
        "multi_scale"
    ]

    print(
        "1. Multi-scale Feature Fusion"
    )

    print(
        f"   Image AUROC : "
        f"{multi_scale['image_auroc']:.4f}"
    )

    print(
        f"   Pixel AUROC : "
        f"{multi_scale['pixel_auroc']:.4f}"
    )

    print(
        f"   Image change: "
        f"+{multi_scale['image_relative_change_percent']:.2f}%"
    )

    print(
        f"   Pixel change: "
        f"+{multi_scale['pixel_relative_change_percent']:.2f}%"
    )

    print()

    adaptive = ablations[
        "adaptive_scoring"
    ]

    print(
        "2. Adaptive Anomaly Scoring"
    )

    print(
        f"   Maximum score AUROC : "
        f"{adaptive['maximum_score_auroc']:.4f}"
    )

    print(
        f"   Adaptive score AUROC: "
        f"{adaptive['adaptive_score_auroc']:.4f}"
    )

    print(
        f"   Relative change     : "
        f"+{adaptive['relative_change_percent']:.2f}%"
    )

    print()

    confidence = ablations[
        "confidence_refinement"
    ]

    print(
        "3. Confidence-aware Refinement"
    )

    print(
        f"   Without refinement : "
        f"{confidence['without_refinement_pixel_auroc']:.4f}"
    )

    print(
        f"   With refinement    : "
        f"{confidence['with_refinement_pixel_auroc']:.4f}"
    )

    print(
        f"   Relative change   : "
        f"{confidence['relative_change_percent']:.2f}%"
    )

    print()

    print("=" * 78)
    print("EXPERIMENTAL CONCLUSIONS")
    print("=" * 78)

    print()

    print(
        "✓ Multi-scale feature fusion showed a strong "
        "positive contribution."
    )

    print(
        "✓ Adaptive anomaly scoring improved "
        "image-level discrimination."
    )

    print(
        "✓ Confidence refinement was tested but did "
        "not improve pixel AUROC."
    )

    print(
        "✓ The final AnomalyVFM+ model achieved "
        "0.9595 Image AUROC."
    )

    print(
        "✓ The final AnomalyVFM+ model achieved "
        "0.8627 Pixel AUROC."
    )

    print()

    print("=" * 78)
    print("FINAL RESULTS COMPLETE")
    print("=" * 78)


# ============================================================
# MAIN
# ============================================================

def main():

    calculate_improvements()

    print_summary()

    save_results()


if __name__ == "__main__":
    main()