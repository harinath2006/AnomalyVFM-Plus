import torch

from models.decoder.scoring import anomaly_score
from models.decoder.calibration import compute_adaptive_score


def main():

    print("=== Old vs Adaptive Anomaly Scoring ===")

    # --------------------------------------------------
    # Create controlled anomaly maps
    # --------------------------------------------------

    anomaly_map = torch.zeros(
        3,
        1,
        224,
        224
    )

    # Case 1:
    # One very strong isolated pixel.
    anomaly_map[0, 0, 100, 100] = 10.0

    # Case 2:
    # A larger moderately anomalous region.
    anomaly_map[1, 0, 80:140, 80:140] = 3.0

    # Case 3:
    # A broad weak anomaly.
    anomaly_map[2] = 1.5

    # --------------------------------------------------
    # Existing max-based score
    # --------------------------------------------------

    max_scores = anomaly_score(
        anomaly_map
    )

    # --------------------------------------------------
    # New adaptive score
    # --------------------------------------------------

    adaptive_scores = compute_adaptive_score(
        anomaly_map
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    for index in range(3):

        print()
        print(
            f"Case {index + 1}"
        )

        print(
            "Max score:",
            max_scores[index].item()
        )

        print(
            "Adaptive score:",
            adaptive_scores[index].item()
        )

    # --------------------------------------------------
    # Basic checks
    # --------------------------------------------------

    assert max_scores.shape == (
        3,
    )

    assert adaptive_scores.shape == (
        3,
    )

    assert torch.all(
        adaptive_scores >= 0
    )

    assert torch.all(
        adaptive_scores <= 1
    )

    print()
    print(
        "Old scoring and adaptive scoring both work."
    )

    print(
        "Scoring comparison test: SUCCESS"
    )


if __name__ == "__main__":
    main()