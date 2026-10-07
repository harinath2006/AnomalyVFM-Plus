from pathlib import Path

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


def get_score(
    model,
    image_path,
):

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


def main():

    print("=" * 70)
    print("IMAGE SCORE SEPARATION TEST")
    print("=" * 70)

    model = load_model()

    good_images = sorted(
        GOOD_DIR.glob("*.png")
    )[:10]

    anomaly_images = sorted(
        ANOMALY_DIR.glob("*.png")
    )[:10]

    if not good_images:
        raise RuntimeError(
            f"No normal images found in {GOOD_DIR}"
        )

    if not anomaly_images:
        raise RuntimeError(
            f"No anomaly images found in {ANOMALY_DIR}"
        )

    good_scores = []
    anomaly_scores = []

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    print()
    print("NORMAL IMAGES")
    print("-" * 70)

    for image_path in good_images:

        score = get_score(
            model,
            image_path,
        )

        good_scores.append(score)

        print(
            f"{image_path.name:30s}"
            f" score={score:.6f}"
        )

    # --------------------------------------------------------
    # ANOMALY
    # --------------------------------------------------------

    print()
    print("ANOMALY IMAGES")
    print("-" * 70)

    for image_path in anomaly_images:

        score = get_score(
            model,
            image_path,
        )

        anomaly_scores.append(score)

        print(
            f"{image_path.name:30s}"
            f" score={score:.6f}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    normal_mean = (
        sum(good_scores)
        /
        len(good_scores)
    )

    anomaly_mean = (
        sum(anomaly_scores)
        /
        len(anomaly_scores)
    )

    print()
    print("=" * 70)

    print(
        f"Normal mean   : {normal_mean:.6f}"
    )

    print(
        f"Anomaly mean  : {anomaly_mean:.6f}"
    )

    print(
        f"Normal min    : {min(good_scores):.6f}"
    )

    print(
        f"Normal max    : {max(good_scores):.6f}"
    )

    print(
        f"Anomaly min   : {min(anomaly_scores):.6f}"
    )

    print(
        f"Anomaly max   : {max(anomaly_scores):.6f}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Separation check
    # --------------------------------------------------------

    if (
        min(anomaly_scores)
        >
        max(good_scores)
    ):

        print(
            "SUCCESS:"
            " Normal and anomaly scores are separated."
        )

    else:

        print(
            "WARNING:"
            " Normal and anomaly scores still overlap."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()