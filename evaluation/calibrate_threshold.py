from pathlib import Path
import json

import torch
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus


CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

GOOD_DIR = Path(
    "data/raw/mvtec/bottle/test/good"
)

ANOMALY_DIR = Path(
    "data/raw/mvtec/bottle/test/broken_large"
)

OUTPUT_PATH = Path(
    "results/evaluation/threshold.json"
)


def load_model():

    print("Loading AnomalyVFM+...")

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print("Checkpoint loaded.")

    return model


def get_score(model, image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    with torch.no_grad():

        _, image_logit = (
            model.forward_with_score(
                [image]
            )
        )

        score = torch.sigmoid(
            image_logit
        )[0].item()

    return score


def collect_scores(
    model,
    image_paths,
    label,
):

    results = []

    for image_path in image_paths:

        score = get_score(
            model,
            image_path,
        )

        results.append(
            {
                "score": score,
                "label": label,
                "path": str(image_path),
            }
        )

    return results


def find_best_threshold(results):

    scores = sorted(
        set(
            item["score"]
            for item in results
        )
    )

    best_threshold = None
    best_f1 = -1.0

    for threshold in scores:

        predictions = []

        labels = []

        for item in results:

            prediction = (
                1
                if item["score"] >= threshold
                else 0
            )

            predictions.append(
                prediction
            )

            labels.append(
                item["label"]
            )

        tp = sum(
            p == 1 and y == 1
            for p, y in zip(
                predictions,
                labels,
            )
        )

        fp = sum(
            p == 1 and y == 0
            for p, y in zip(
                predictions,
                labels,
            )
        )

        fn = sum(
            p == 0 and y == 1
            for p, y in zip(
                predictions,
                labels,
            )
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        if (
            precision + recall
            > 0
        ):

            f1 = (
                2
                * precision
                * recall
                /
                (precision + recall)
            )

        else:

            f1 = 0.0

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def main():

    print("=" * 70)
    print("ANOMALYVFM+ THRESHOLD CALIBRATION")
    print("=" * 70)

    model = load_model()

    good_images = sorted(
        GOOD_DIR.glob("*.png")
    )

    anomaly_images = sorted(
        ANOMALY_DIR.glob("*.png")
    )

    if len(good_images) < 20:
        raise RuntimeError(
            "Need at least 20 normal images."
        )

    if len(anomaly_images) < 20:
        raise RuntimeError(
            "Need at least 20 anomaly images."
        )

    # --------------------------------------------------------
    # Calibration set
    #
    # First 10 normal + first 10 anomaly
    # --------------------------------------------------------

    calibration_good = good_images[:10]

    calibration_anomaly = anomaly_images[:10]

    calibration_results = []

    calibration_results.extend(
        collect_scores(
            model,
            calibration_good,
            label=0,
        )
    )

    calibration_results.extend(
        collect_scores(
            model,
            calibration_anomaly,
            label=1,
        )
    )

    # --------------------------------------------------------
    # Find threshold
    # --------------------------------------------------------

    threshold, f1 = find_best_threshold(
        calibration_results
    )

    print()
    print("CALIBRATION RESULTS")
    print("-" * 70)

    print(
        f"Threshold : {threshold:.6f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Save threshold
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "threshold": float(
                    threshold
                ),
                "calibration_f1": float(
                    f1
                ),
                "calibration_normal_images": 10,
                "calibration_anomaly_images": 10,
                "score_source": "image_predictor",
            },
            file,
            indent=4,
        )

    print()
    print(
        "Threshold saved:",
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Held-out test
    #
    # Remaining images are NOT used to choose threshold.
    # --------------------------------------------------------

    test_good = good_images[10:30]

    test_anomaly = anomaly_images[10:30]

    test_results = []

    test_results.extend(
        collect_scores(
            model,
            test_good,
            label=0,
        )
    )

    test_results.extend(
        collect_scores(
            model,
            test_anomaly,
            label=1,
        )
    )

    correct = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in test_results:

        prediction = (
            1
            if item["score"] >= threshold
            else 0
        )

        label = item["label"]

        if prediction == label:
            correct += 1

        if (
            prediction == 1
            and label == 1
        ):
            tp += 1

        elif (
            prediction == 0
            and label == 0
        ):
            tn += 1

        elif (
            prediction == 1
            and label == 0
        ):
            fp += 1

        elif (
            prediction == 0
            and label == 1
        ):
            fn += 1

    total = len(test_results)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("HELD-OUT RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print()
    print("Confusion Matrix")
    print(
        f"TN={tn}  FP={fp}"
    )
    print(
        f"FN={fn}  TP={tp}"
    )

    print("=" * 70)
    print("THRESHOLD CALIBRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()