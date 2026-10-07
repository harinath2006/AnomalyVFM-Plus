import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULT_DIR = Path("results/evaluation")
OUTPUT_DIR = Path("results/plots")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_result(filename):
    path = RESULT_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("=== AnomalyVFM+ Result Visualization ===")

    baseline = load_result("baseline.json")
    lora = load_result("lora.json")
    plus = load_result("anomalyvfm_plus.json")

    models = [
        "Baseline",
        "LoRA",
        "AnomalyVFM+"
    ]

    image_auroc = [
        baseline["image_auroc"],
        lora["image_auroc"],
        plus["image_auroc"]
    ]

    pixel_auroc = [
        baseline["pixel_auroc"],
        lora["pixel_auroc"],
        plus["pixel_auroc"]
    ]

    print()
    print("Image AUROC:")
    for model, value in zip(models, image_auroc):
        print(f"{model}: {value:.4f}")

    print()
    print("Pixel AUROC:")
    for model, value in zip(models, pixel_auroc):
        print(f"{model}: {value:.4f}")

    # --------------------------------------------------
    # Image-level AUROC
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        image_auroc
    )

    plt.title("Image-Level AUROC Comparison")
    plt.ylabel("Image AUROC")
    plt.ylim(0, 1)

    for bar, value in zip(bars, image_auroc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    image_path = OUTPUT_DIR / "image_auroc_comparison.png"

    plt.savefig(
        image_path,
        dpi=300
    )

    plt.close()

    print()
    print("Saved:", image_path)

    # --------------------------------------------------
    # Pixel-level AUROC
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        models,
        pixel_auroc
    )

    plt.title("Pixel-Level AUROC Comparison")
    plt.ylabel("Pixel AUROC")
    plt.ylim(0, 1)

    for bar, value in zip(bars, pixel_auroc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.4f}",
            ha="center"
        )

    plt.tight_layout()

    pixel_path = OUTPUT_DIR / "pixel_auroc_comparison.png"

    plt.savefig(
        pixel_path,
        dpi=300
    )

    plt.close()

    print("Saved:", pixel_path)

    # --------------------------------------------------
    # Combined comparison
    # --------------------------------------------------

    x = range(len(models))
    width = 0.35

    plt.figure(figsize=(9, 5))

    image_bars = plt.bar(
        [i - width / 2 for i in x],
        image_auroc,
        width,
        label="Image AUROC"
    )

    pixel_bars = plt.bar(
        [i + width / 2 for i in x],
        pixel_auroc,
        width,
        label="Pixel AUROC"
    )

    plt.xticks(
        list(x),
        models
    )

    plt.ylabel("AUROC")
    plt.title("Anomaly Detection Performance Comparison")
    plt.ylim(0, 1)

    plt.legend()

    for bar, value in zip(image_bars, image_auroc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

    for bar, value in zip(pixel_bars, pixel_auroc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    combined_path = OUTPUT_DIR / "model_comparison.png"

    plt.savefig(
        combined_path,
        dpi=300
    )

    plt.close()

    print("Saved:", combined_path)

    print()
    print("=== RESULT VISUALIZATION: SUCCESS ===")


if __name__ == "__main__":
    main()