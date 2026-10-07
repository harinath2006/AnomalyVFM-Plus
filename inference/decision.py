import torch


def calculate_spatial_evidence(
    anomaly_map,
    map_threshold=0.60,
):
    """
    Calculate how much of the image contains strong anomaly evidence.

    Parameters
    ----------
    anomaly_map : torch.Tensor
        Anomaly map with shape [H, W], [1, H, W],
        [B, 1, H, W], or [B, H, W].

    map_threshold : float
        Pixel-level probability threshold.

    Returns
    -------
    float
        Fraction of pixels considered anomalous.
    """

    anomaly_map = torch.as_tensor(anomaly_map).detach().float()

    # Convert logits to probabilities if necessary.
    probabilities = torch.sigmoid(anomaly_map)

    # Remove unnecessary dimensions.
    probabilities = probabilities.squeeze()

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a single 2D anomaly map after squeezing, "
            f"got shape {tuple(probabilities.shape)}"
        )

    anomalous_pixels = probabilities >= map_threshold

    anomalous_ratio = anomalous_pixels.float().mean().item()

    return anomalous_ratio


def make_decision(
    anomaly_score,
    anomaly_map,
    score_threshold,
    map_threshold=0.60,
    spatial_ratio_threshold=0.01,
):
    """
    Make the final NORMAL / ANOMALOUS decision.

    The decision uses two sources of evidence:

    1. Image-level anomaly score.
    2. Spatial anomaly evidence from the anomaly map.

    The image is classified as anomalous when either:

        anomaly score >= score threshold

    OR

        anomalous pixel ratio >= spatial ratio threshold
        AND the anomaly map contains sufficiently strong evidence.

    Returns
    -------
    dict
        Complete decision information.
    """

    score = float(torch.as_tensor(anomaly_score).item())

    spatial_ratio = calculate_spatial_evidence(
        anomaly_map,
        map_threshold=map_threshold,
    )

    score_evidence = score >= score_threshold
    spatial_evidence = spatial_ratio >= spatial_ratio_threshold

    is_anomalous = score_evidence or spatial_evidence

    if is_anomalous:
        prediction = "ANOMALOUS"
    else:
        prediction = "NORMAL"

    if score_evidence and spatial_evidence:
        reason = "strong image-level and spatial anomaly evidence"
    elif score_evidence:
        reason = "image-level anomaly score exceeded threshold"
    elif spatial_evidence:
        reason = "spatial anomaly evidence exceeded threshold"
    else:
        reason = "insufficient anomaly evidence"

    return {
        "prediction": prediction,
        "anomaly_score": score,
        "score_threshold": float(score_threshold),
        "anomalous_pixel_ratio": spatial_ratio,
        "map_threshold": float(map_threshold),
        "spatial_ratio_threshold": float(spatial_ratio_threshold),
        "score_evidence": score_evidence,
        "spatial_evidence": spatial_evidence,
        "reason": reason,
    }


def print_decision(result):
    """
    Print a clean final inference result.
    """

    print()
    print("=" * 60)
    print("FINAL ANOMALY DECISION")
    print("=" * 60)

    print(f"Prediction             : {result['prediction']}")
    print(f"Anomaly score          : {result['anomaly_score']:.4f}")
    print(f"Score threshold        : {result['score_threshold']:.4f}")
    print(
        f"Anomalous pixel ratio  : "
        f"{result['anomalous_pixel_ratio']:.4f}"
    )
    print(f"Map threshold          : {result['map_threshold']:.4f}")
    print(
        f"Spatial ratio threshold: "
        f"{result['spatial_ratio_threshold']:.4f}"
    )

    print()
    print(
        "Score evidence         : "
        f"{'YES' if result['score_evidence'] else 'NO'}"
    )

    print(
        "Spatial evidence       : "
        f"{'YES' if result['spatial_evidence'] else 'NO'}"
    )

    print(f"Decision reason        : {result['reason']}")

    print("=" * 60)


def main():
    """
    Basic module test.
    """

    print("=== AnomalyVFM+ Decision Module Test ===")

    # Example anomaly logits.
    anomaly_map = torch.full(
        (224, 224),
        -4.0,
        dtype=torch.float32,
    )

    # Create a synthetic strong anomaly region.
    anomaly_map[90:140, 90:140] = 4.0

    anomaly_score = torch.tensor(0.20)

    result = make_decision(
        anomaly_score=anomaly_score,
        anomaly_map=anomaly_map,
        score_threshold=0.2353,
        map_threshold=0.60,
        spatial_ratio_threshold=0.01,
    )

    print_decision(result)

    assert result["prediction"] == "ANOMALOUS"
    assert result["anomalous_pixel_ratio"] > 0.01
    assert result["spatial_evidence"] is True

    print()
    print("=== DECISION MODULE: SUCCESS ===")


if __name__ == "__main__":
    main()