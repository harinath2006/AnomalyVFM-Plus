import torch
from sklearn.metrics import roc_auc_score


def image_level_auroc(scores, labels):
    """
    Compute image-level AUROC.

    Args:
        scores: [N] anomaly scores.
        labels: [N] labels where 0 = normal and 1 = anomalous.

    Returns:
        AUROC as a float in [0, 1].
    """
    scores = (
        torch.as_tensor(scores)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels = (
        torch.as_tensor(labels)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    if len(scores) != len(labels):
        raise ValueError(
            "Image scores and labels must have equal lengths."
        )

    if len(scores) == 0:
        raise ValueError(
            "Cannot calculate image AUROC on an empty dataset."
        )

    if len(set(labels.tolist())) < 2:
        raise ValueError(
            "Image AUROC requires both normal (0) and anomalous (1) images."
        )

    return float(roc_auc_score(labels, scores))


def pixel_level_auroc(anomaly_maps, masks):
    """
    Compute pixel-level AUROC.

    Args:
        anomaly_maps: anomaly scores shaped [N, H, W] or [N, 1, H, W].
        masks: ground-truth masks with the same shape; 0 = normal pixel,
            nonzero = anomalous pixel.

    Returns:
        Pixel AUROC as a float in [0, 1].
    """
    anomaly_maps = (
        torch.as_tensor(anomaly_maps)
        .detach()
        .cpu()
        .float()
    )

    masks = (
        torch.as_tensor(masks)
        .detach()
        .cpu()
    )

    if anomaly_maps.shape != masks.shape:
        raise ValueError(
            "Anomaly maps and masks must have identical shapes. "
            f"Got {tuple(anomaly_maps.shape)} and {tuple(masks.shape)}."
        )

    if anomaly_maps.numel() == 0:
        raise ValueError(
            "Cannot calculate pixel AUROC on empty inputs."
        )

    anomaly_maps = anomaly_maps.flatten().numpy()
    masks = (masks.flatten().numpy() > 0).astype("uint8")

    if len(set(masks.tolist())) < 2:
        raise ValueError(
            "Pixel AUROC requires both normal and anomalous pixels "
            "across the evaluated masks."
        )

    return float(roc_auc_score(masks, anomaly_maps))


if __name__ == "__main__":
    print("=== Evaluation Metrics Test ===")

    image_scores = torch.tensor([0.10, 0.20, 0.80, 0.90])
    image_labels = torch.tensor([0, 0, 1, 1])

    image_auc = image_level_auroc(image_scores, image_labels)
    print(f"Image AUROC: {image_auc:.4f}")
    assert 0.0 <= image_auc <= 1.0

    anomaly_maps = torch.tensor(
        [[[[0.1, 0.2], [0.8, 0.9]]]]
    )
    masks = torch.tensor(
        [[[[0, 0], [1, 1]]]]
    )

    pixel_auc = pixel_level_auroc(anomaly_maps, masks)
    print(f"Pixel AUROC: {pixel_auc:.4f}")
    assert 0.0 <= pixel_auc <= 1.0

    print()
    print("Evaluation metrics test: SUCCESS")
