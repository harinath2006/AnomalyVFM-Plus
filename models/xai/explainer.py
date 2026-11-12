import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import matplotlib


def _spatial_quadrant(norm_x: float, norm_y: float) -> str:
    if 0.35 <= norm_x <= 0.65 and 0.35 <= norm_y <= 0.65:
        return "Central Region"
    v = "Upper" if norm_y < 0.38 else ("Lower" if norm_y > 0.62 else "Middle")
    h = "Left" if norm_x < 0.38 else ("Right" if norm_x > 0.62 else "Center")
    return f"{v}-{h} Quadrant"


def _connected_components_count(binary_mask: np.ndarray) -> int:
    """Pure-NumPy 4-connected component counter (no OpenCV/SciPy dependency needed)."""
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(binary_mask > 0)
    count = 0
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        count += 1
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and binary_mask[ny, nx] > 0:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


@torch.no_grad()
def explain_prediction(
    model,
    pil_image: Image.Image,
    anomaly_map: torch.Tensor,
    image_score: float,
    threshold: float,
    output_size: tuple[int, int] | None = None,
) -> dict:
    """
    Generates multi-scale DINOv2 layer attribution, spatial hotspot metrics,
    an annotated contour/bounding-box overlay, and a plain-English XAI report.
    """
    orig_w, orig_h = pil_image.size
    out_w, out_h = output_size if output_size is not None else (orig_w, orig_h)

    # 1. Upsample probability map to target visualization size
    prob_tensor = torch.sigmoid(anomaly_map.detach().float().cpu())
    prob_resized = F.interpolate(
        prob_tensor,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    prob_resized = np.clip(prob_resized, 0.0, 1.0)

    # 2. Foreground ROI mask
    roi_tensor = model.compute_foreground_roi([pil_image], target_size=out_h)
    roi_resized = F.interpolate(
        roi_tensor.detach().float().cpu(),
        size=(out_h, out_w),
        mode="nearest",
    )[0, 0].numpy()
    roi_bool = roi_resized > 0.5
    roi_pixel_count = max(1, int(roi_bool.sum()))

    # 3. Multi-scale DINOv2 Layer Attribution (Layers 3, 6 vs Layers 9, 12)
    multiscale_feats = model.extract_multiscale_features([pil_image])
    layer_energies = []
    for feat in multiscale_feats:
        # L2 norm per spatial patch -> upsample -> measure energy inside hotspot vs background
        spatial_norm = torch.norm(feat.detach().float().cpu(), dim=1, keepdim=True)
        spatial_norm = F.interpolate(spatial_norm, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0].numpy()
        layer_energies.append(spatial_norm)

    # Adaptive hotspot threshold within ROI
    roi_probs = prob_resized[roi_bool] if roi_bool.any() else prob_resized.flatten()
    peak_conf = float(roi_probs.max())
    mean_roi_conf = float(roi_probs.mean())
    active_thresh = max(0.5, float(np.percentile(roi_probs, 92)))
    hotspot_mask = ((prob_resized >= active_thresh) & roi_bool).astype(np.uint8)

    # Compute shallow (layers 3+6: texture/edges) vs deep (layers 9+12: semantics/structure) attribution
    if hotspot_mask.sum() > 0:
        shallow_ratio = (
            (layer_energies[0][hotspot_mask > 0].mean() / (layer_energies[0][roi_bool].mean() + 1e-6))
            + (layer_energies[1][hotspot_mask > 0].mean() / (layer_energies[1][roi_bool].mean() + 1e-6))
        ) * 0.5
        deep_ratio = (
            (layer_energies[2][hotspot_mask > 0].mean() / (layer_energies[2][roi_bool].mean() + 1e-6))
            + (layer_energies[3][hotspot_mask > 0].mean() / (layer_energies[3][roi_bool].mean() + 1e-6))
        ) * 0.5
    else:
        shallow_ratio, deep_ratio = 1.0, 1.0

    total_attr = float(max(1e-6, shallow_ratio + deep_ratio))
    texture_pct = round(float(shallow_ratio / total_attr) * 100.0, 1)
    structure_pct = round(float(deep_ratio / total_attr) * 100.0, 1)

    # 4. Spatial & Morphological Hotspot Analysis
    affected_pixels = int(hotspot_mask.sum())
    affected_area_pct = round((affected_pixels / roi_pixel_count) * 100.0, 2)
    region_count = _connected_components_count(hotspot_mask)

    if affected_pixels > 0:
        ys, xs = np.where(hotspot_mask > 0)
        weights = prob_resized[ys, xs]
        weights_sum = float(weights.sum()) + 1e-6
        cx = float((xs * weights).sum() / weights_sum)
        cy = float((ys * weights).sum() / weights_sum)
        location_label = _spatial_quadrant(cx / out_w, cy / out_h)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        max_idx = np.unravel_index(np.argmax(prob_resized * roi_bool), prob_resized.shape)
        cy, cx = float(max_idx[0]), float(max_idx[1])
        location_label = _spatial_quadrant(cx / out_w, cy / out_h)
        bbox = None

    # Focus / sharpness classification
    if affected_area_pct < 8.0 and peak_conf >= 0.65:
        pattern_type = "Focal / Localized Hotspot"
    elif affected_area_pct >= 8.0:
        pattern_type = "Diffuse / Extended Region"
    else:
        pattern_type = "Low-Activation Background Uniformity"

    # 5. Build Annotated XAI Image (Heatmap Overlay + ROI Contour + Hotspot Bounding Box)
    display_img = pil_image.resize((out_w, out_h), Image.Resampling.LANCZOS)
    colored_heatmap = (matplotlib.colormaps["jet"](prob_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored_heatmap)
    blended = Image.blend(display_img, heatmap_img, alpha=0.42)

    draw = ImageDraw.Draw(blended)
    # Draw hotspot contour edges in bright cyan
    padded = np.pad(hotspot_mask, 1, mode="constant")
    edges = (
        (hotspot_mask == 1)
        & (
            (padded[:-2, 1:-1] == 0)
            | (padded[2:, 1:-1] == 0)
            | (padded[1:-1, :-2] == 0)
            | (padded[1:-1, 2:] == 0)
        )
    )
    ey, ex = np.where(edges)
    for px, py in zip(ex.tolist(), ey.tolist()):
        draw.point((px, py), fill=(0, 255, 255))

    # Draw bounding box & peak crosshair if anomalous / focal hotspot exists
    is_anomalous = image_score >= threshold
    if bbox is not None and (is_anomalous or peak_conf >= 0.55):
        x1, y1, x2, y2 = bbox
        box_color = (255, 60, 60) if is_anomalous else (255, 215, 0)
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        icx, icy = int(cx), int(cy)
        draw.line([(max(0, icx - 6), icy), (min(out_w - 1, icx + 6), icy)], fill=(255, 255, 255), width=2)
        draw.line([(icx, max(0, icy - 6)), (icx, min(out_h - 1, icy + 6))], fill=(255, 255, 255), width=2)

    # 6. Plain-English Diagnostic Narrative
    dominant_cue = (
        "fine surface texture and edge disruptions (DINOv2 shallow layers 3–6)"
        if texture_pct >= structure_pct
        else "high-level structural and morphological deviations (DINOv2 deep layers 9–12)"
    )

    if is_anomalous:
        summary = (
            f"Anomalous activation detected primarily in the {location_label} "
            f"({pattern_type}, covering {affected_area_pct}% of foreground tissue across "
            f"{region_count} distinct region(s)). Peak spatial confidence reached "
            f"{peak_conf * 100:.1f}% (ROI baseline mean: {mean_roi_conf * 100:.1f}%). "
            f"Feature attribution indicates the prediction is driven {max(texture_pct, structure_pct):.1f}% by {dominant_cue}."
        )
    else:
        summary = (
            f"No significant structural or textural anomaly detected above the decision threshold "
            f"({threshold:.4f}). Foreground tissue exhibits uniform activation "
            f"(mean ROI score: {mean_roi_conf * 100:.1f}%, peak: {peak_conf * 100:.1f}% in {location_label}), "
            f"consistent with normal baseline appearance."
        )

    return {
        "heatmap_image": heatmap_img,
        "overlay_image": Image.blend(display_img, heatmap_img, alpha=0.45),
        "xai_annotated_image": blended,
        "xai_report": {
            "summary": summary,
            "location": location_label,
            "pattern_type": pattern_type,
            "affected_area_percent": affected_area_pct,
            "peak_confidence_percent": round(peak_conf * 100.0, 2),
            "mean_roi_confidence_percent": round(mean_roi_conf * 100.0, 2),
            "region_count": region_count,
            "layer_attribution": {
                "shallow_texture_percent": texture_pct,
                "deep_structure_percent": structure_pct,
            },
        },
    }