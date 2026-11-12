from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from sklearn.metrics import roc_auc_score

from models.anomaly_vfm_plus import AnomalyVFMPlus


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path(
    "data/raw/mvtec/bottle/test"
)

CHECKPOINTS = {
    "V1": Path(
        "experiments/plus/checkpoint.pt"
    ),

    "V2": Path(
        "experiments/plus_v2/checkpoint.pt"
    ),
}

DEVICE = torch.device("cpu")

IMAGE_SIZE = 224


# ============================================================
# COLLECT TEST SAMPLES
# ============================================================

def collect_samples():

    samples = []

    good_dir = TEST_DIR / "good"

    for path in sorted(good_dir.glob("*.png")):

        samples.append({
            "path": path,
            "label": 0,
        })

    for category_dir in sorted(TEST_DIR.iterdir()):

        if not category_dir.is_dir():
            continue

        if category_dir.name == "good":
            continue

        for path in sorted(category_dir.glob("*.png")):

            samples.append({
                "path": path,
                "label": 1,
            })

    return samples


# ============================================================
# GENERATE IMAGE-LEVEL SCORES
# ============================================================

def get_aggregation_scores(anomaly_map):

    flat = anomaly_map.flatten()

    sorted_scores = torch.sort(
        flat,
        descending=True,
    ).values

    total = sorted_scores.numel()

    scores = {}

    # Maximum pixel score

    scores["MAX"] = sorted_scores[0].item()

    # Top 1%

    k1 = max(
        1,
        int(total * 0.01),
    )

    scores["TOP_1_PERCENT"] = (
        sorted_scores[:k1].mean().item()
    )

    # Top 5%

    k5 = max(
        1,
        int(total * 0.05),
    )

    scores["TOP_5_PERCENT"] = (
        sorted_scores[:k5].mean().item()
    )

    # Top 10%

    k10 = max(
        1,
        int(total * 0.10),
    )

    scores["TOP_10_PERCENT"] = (
        sorted_scores[:k10].mean().item()
    )

    # Top 20%

    k20 = max(
        1,
        int(total * 0.20),
    )

    scores["TOP_20_PERCENT"] = (
        sorted_scores[:k20].mean().item()
    )

    # Overall mean

    scores["MEAN"] = flat.mean().item()

    return scores


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    model,
    checkpoint_path,
    samples,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    aggregation_results = {}

    image_predictor_scores = []

    labels = []

    for method in [
        "MAX",
        "TOP_1_PERCENT",
        "TOP_5_PERCENT",
        "TOP_10_PERCENT",
        "TOP_20_PERCENT",
        "MEAN",
    ]:

        aggregation_results[method] = []

    with torch.inference_mode():

        for index, sample in enumerate(
            samples,
            start=1,
        ):

            image = Image.open(
                sample["path"]
            ).convert("RGB")

            anomaly_map, image_logit = (
                model.forward_with_score(
                    [image]
                )
            )

            image_score = torch.sigmoid(
                image_logit
            ).reshape(-1)[0].item()

            # Convert decoder logits into probabilities.

            anomaly_map = torch.sigmoid(
                anomaly_map
            )

            # Resize to evaluation resolution.

            anomaly_map = F.interpolate(
                anomaly_map,
                size=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                mode="bilinear",
                align_corners=False,
            )

            anomaly_map = anomaly_map[0, 0]

            image_predictor_scores.append(
                image_score
            )

            map_scores = get_aggregation_scores(
                anomaly_map
            )

            for method, score in map_scores.items():

                aggregation_results[method].append(
                    score
                )

            labels.append(
                sample["label"]
            )

            if index % 25 == 0:

                print(
                    f"Processed {index}/{len(samples)}"
                )

    labels = np.asarray(labels)

    results = {}

    # Original image predictor

    results["IMAGE_PREDICTOR"] = roc_auc_score(
        labels,
        image_predictor_scores,
    )

    # Pixel-map aggregation methods

    for method, scores in aggregation_results.items():

        results[method] = roc_auc_score(
            labels,
            scores,
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("ANOMALYVFM+ IMAGE SCORING EXPERIMENT")
    print("=" * 75)

    samples = collect_samples()

    print("Total images:", len(samples))

    print("Normal:", sum(
        s["label"] == 0
        for s in samples
    ))

    print("Anomalous:", sum(
        s["label"] == 1
        for s in samples
    ))

    print()
    print("Loading model...")

    model = AnomalyVFMPlus()

    model.to(DEVICE)

    all_results = {}

    for name, checkpoint_path in CHECKPOINTS.items():

        print()
        print("=" * 75)
        print("EVALUATING", name)
        print("=" * 75)

        results = evaluate_checkpoint(
            model,
            checkpoint_path,
            samples,
        )

        all_results[name] = results

        for method, auc in results.items():

            print(
                f"{method:20s}: {auc:.4f}"
            )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print("=" * 75)
    print("FINAL IMAGE-LEVEL SCORING COMPARISON")
    print("=" * 75)

    methods = all_results["V1"].keys()

    print(
        f"{'Method':22s} {'V1':>10s} {'V2':>10s}"
    )

    print("-" * 50)

    for method in methods:

        v1 = all_results["V1"][method]

        v2 = all_results["V2"][method]

        print(
            f"{method:22s} "
            f"{v1:10.4f} "
            f"{v2:10.4f}"
        )

    print("=" * 75)


if __name__ == "__main__":

    main()