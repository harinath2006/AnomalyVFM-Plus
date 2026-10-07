import torch
from sklearn.metrics import roc_auc_score


def image_level_auroc(scores, labels):
    """
    Compute image-level AUROC.

    scores:
        [N] anomaly scores

    labels:
        [N] where 0 = normal, 1 = anomalous
    """

    scores = torch.as_tensor(scores).detach().cpu().numpy()
    labels = torch.as_tensor(labels).detach().cpu().numpy()

    return roc_auc_score(labels, scores)


def pixel_level_auroc(anomaly_maps, masks):
    """
    Compute pixel-level AUROC.

    anomaly_maps:
        [N, 1, H, W]

    masks:
        [N, 1, H, W]
        0 = normal pixel
        1 = anomalous pixel
    """

    anomaly_maps = torch.as_tensor(anomaly_maps).detach().cpu()
    masks = torch.as_tensor(masks).detach().cpu()

    anomaly_maps = anomaly_maps.flatten().numpy()
    masks = masks.flatten().numpy()

    return roc_auc_score(masks, anomaly_maps)


if __name__ == "__main__":

    print("=== Evaluation Metrics Test ===")

    # ------------------------------------------------
    # Image-level test
    # ------------------------------------------------

    scores = torch.tensor([
        0.10,
        0.20,
        0.80,
        0.90
    ])

    labels = torch.tensor([
        0,
        0,
        1,
        1
    ])

    image_auc = image_level_auroc(
        scores,
        labels
    )

    print("Image AUROC:", image_auc)

    assert 0.0 <= image_auc <= 1.0

    # ------------------------------------------------
    # Pixel-level test
    # ------------------------------------------------

    anomaly_maps = torch.tensor([
        [
            [
                [0.1, 0.2],
                [0.8, 0.9]
            ]
        ]
    ])

    masks = torch.tensor([
        [
            [
                [0, 0],
                [1, 1]
            ]
        ]
    ])

    pixel_auc = pixel_level_auroc(
        anomaly_maps,
        masks
    )

    print("Pixel AUROC:", pixel_auc)

    assert 0.0 <= pixel_auc <= 1.0

    print()
    print("Evaluation metrics test: SUCCESS")