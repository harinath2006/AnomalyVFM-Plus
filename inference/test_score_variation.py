from pathlib import Path

from PIL import Image

from inference.predict import load_model, predict_image


# ============================================================
# TEST IMAGES
# ============================================================

TEST_IMAGES = [
    Path("data/raw/mvtec/bottle/test/good/000.png"),
    Path("data/raw/mvtec/bottle/test/good/001.png"),
    Path("data/raw/mvtec/bottle/test/good/002.png"),
    Path("data/raw/mvtec/bottle/test/broken_large/000.png"),
    Path("data/raw/mvtec/bottle/test/broken_small/000.png"),
    Path("data/raw/mvtec/bottle/test/contamination/000.png"),
]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("ANOMALYVFM+ SCORE VARIATION DIAGNOSTIC")
    print("=" * 70)

    print()
    print("Loading model...")

    model = load_model()

    print("Model loaded successfully.")

    print()
    print("Testing images...")
    print()

    scores = []

    for image_path in TEST_IMAGES:

        if not image_path.exists():

            print(
                "MISSING:",
                image_path
            )

            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Basic image statistics
        # ----------------------------------------------------

        image_pixels = list(
            image.getdata()
        )

        mean_pixel = sum(
            sum(pixel)
            for pixel in image_pixels
        ) / (
            len(image_pixels) * 3
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        result = predict_image(
            model,
            image
        )

        score = float(
            result["anomaly_score"]
        )

        scores.append(score)

        print(
            f"{image_path.name:<12} "
            f"{image_path.parent.name:<18} "
            f"mean_pixel={mean_pixel:8.3f} "
            f"score={score:.8f}"
        )

    # ========================================================
    # SCORE ANALYSIS
    # ========================================================

    print()
    print("-" * 70)

    if len(scores) > 0:

        minimum = min(scores)
        maximum = max(scores)

        print(
            f"Minimum score : {minimum:.8f}"
        )

        print(
            f"Maximum score : {maximum:.8f}"
        )

        print(
            f"Score range   : {maximum - minimum:.8f}"
        )

        print(
            f"Unique scores : {len(set(scores))}"
        )

    print("-" * 70)

    # ========================================================
    # DIAGNOSIS
    # ========================================================

    if len(scores) >= 2:

        score_range = max(scores) - min(scores)

        print()

        if score_range < 0.001:

            print(
                "WARNING:"
            )

            print(
                "All scores are almost identical."
            )

            print(
                "The inference pipeline needs debugging."
            )

        else:

            print(
                "Scores vary across images."
            )

            print(
                "Inference pipeline is responding to input."
            )

    print()
    print("=" * 70)
    print("=== SCORE VARIATION TEST COMPLETE ===")
    print("=" * 70)


if __name__ == "__main__":
    main()