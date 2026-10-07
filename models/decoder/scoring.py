import torch


def anomaly_score(anomaly_map):
    """
    Convert an anomaly map into a single anomaly score.

    Input:
        anomaly_map: [B, 1, H, W]

    Output:
        score: [B]
    """

    probabilities = torch.sigmoid(anomaly_map)

    score = probabilities.flatten(1).max(dim=1).values

    return score