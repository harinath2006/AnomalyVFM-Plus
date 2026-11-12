import torch


def compute_adaptive_score(
    anomaly_map,
    percentile=0.985,
    roi_mask=None,
):
    """
    Compute ROI-aware adaptive anomaly score.
    Focuses on top-percentile activations within actual foreground tissue.

    Input:
        anomaly_map: [B, 1, H, W] logits
        roi_mask: optional [B, 1, H, W] binary foreground mask (1 = tissue, 0 = background)
    Output:
        score: [B]
    """
    probabilities = torch.sigmoid(anomaly_map)

    if roi_mask is not None:
        if roi_mask.shape[-2:] != probabilities.shape[-2:]:
            roi_mask = torch.nn.functional.interpolate(
                roi_mask.float(),
                size=probabilities.shape[-2:],
                mode="nearest",
            )
        scores = []
        for b in range(probabilities.shape[0]):
            valid = probabilities[b][roi_mask[b] > 0.5]
            if valid.numel() == 0:
                valid = probabilities[b].flatten()
            p_val = torch.quantile(valid, percentile)
            top_k = torch.topk(valid, k=max(1, int(valid.numel() * 0.02))).values.mean()
            scores.append(0.35 * top_k + 0.65 * p_val)
        return torch.stack(scores)

    flattened = probabilities.flatten(1)
    k = max(1, int(flattened.shape[1] * 0.02))
    top_k_mean = torch.topk(flattened, k=k, dim=1).values.mean(dim=1)
    percentile_score = torch.quantile(flattened, percentile, dim=1)

    return 0.35 * top_k_mean + 0.65 * percentile_score