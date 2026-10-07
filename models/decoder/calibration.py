import torch


def compute_adaptive_score(
    anomaly_map,
    percentile=0.95
):
    """
    Compute an adaptive anomaly score.

    The score combines:
    - mean anomaly probability
    - high-percentile anomaly probability

    Input:
        anomaly_map: [B, 1, H, W]

    Output:
        score: [B]
    """

    probabilities = torch.sigmoid(
        anomaly_map
    )

    flattened = probabilities.flatten(1)

    mean_score = flattened.mean(
        dim=1
    )

    percentile_score = torch.quantile(
        flattened,
        percentile,
        dim=1
    )

    score = (
        0.4 * mean_score
        +
        0.6 * percentile_score
    )

    return score


if __name__ == "__main__":

    print(
        "=== Adaptive Score Test ==="
    )

    anomaly_map = torch.randn(
        2,
        1,
        224,
        224
    )

    scores = compute_adaptive_score(
        anomaly_map
    )

    print(
        "Anomaly map:",
        anomaly_map.shape
    )

    print(
        "Scores:",
        scores
    )

    print(
        "Score shape:",
        scores.shape
    )

    assert scores.shape == (
        2,
    )

    assert torch.all(
        scores >= 0
    )

    assert torch.all(
        scores <= 1
    )

    print(
        "Adaptive score test: SUCCESS"
    )