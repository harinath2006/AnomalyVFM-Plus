import torch

from models.decoder.confidence_refinement import refine_anomaly_map
from models.decoder.calibration import compute_adaptive_score


def main():

    print("=== Confidence + Adaptive Scoring Test ===")

    # --------------------------------------------------
    # Synthetic anomaly and confidence maps
    # --------------------------------------------------

    anomaly_map = torch.rand(
        2,
        1,
        224,
        224
    )

    confidence_map = torch.rand(
        2,
        1,
        224,
        224
    )

    # --------------------------------------------------
    # Confidence refinement
    # --------------------------------------------------

    refined_map = refine_anomaly_map(
        anomaly_map,
        confidence_map
    )

    print(
        "Original map:",
        anomaly_map.shape
    )

    print(
        "Confidence map:",
        confidence_map.shape
    )

    print(
        "Refined map:",
        refined_map.shape
    )

    # --------------------------------------------------
    # Scores before and after refinement
    # --------------------------------------------------

    original_scores = compute_adaptive_score(
        anomaly_map
    )

    refined_scores = compute_adaptive_score(
        refined_map
    )

    print(
        "Original scores:",
        original_scores
    )

    print(
        "Confidence-aware scores:",
        refined_scores
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    assert refined_map.shape == anomaly_map.shape

    assert refined_scores.shape == (
        2,
    )

    assert torch.all(
        refined_scores >= 0
    )

    assert torch.all(
        refined_scores <= 1
    )

    # Confidence weighting cannot increase
    # individual anomaly-map values.
    assert torch.all(
        refined_map <= anomaly_map
    )

    print(
        "Confidence refinement: PASS"
    )

    print(
        "Adaptive scoring: PASS"
    )

    print(
        "Confidence-aware scoring test: SUCCESS"
    )


if __name__ == "__main__":
    main()