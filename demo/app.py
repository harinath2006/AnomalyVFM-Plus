from pathlib import Path
import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.xai.explainer import explain_prediction
from inference.threshold import load_threshold


CHECKPOINT_PATH = Path("experiments/plus/checkpoint.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AnomalyVFMPlus()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
model.load_state_dict(state_dict, strict=False)
model = model.to(DEVICE)
model.eval()

THRESHOLD = load_threshold()


def predict(image):
    if image is None:
        return "Please upload an image.", "", None, None, None

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB")

    with torch.no_grad():
        anomaly_map, image_logit = model.forward_with_score([image])
        score = torch.sigmoid(image_logit).reshape(-1)[0].item()

    xai_data = explain_prediction(
        model=model,
        pil_image=image,
        anomaly_map=anomaly_map,
        image_score=score,
        threshold=THRESHOLD,
    )
    xai = xai_data["xai_report"]
    attr = xai["layer_attribution"]

    prediction = "ANOMALOUS" if score >= THRESHOLD else "NORMAL"
    result_text = (
        f"## {prediction}\n\n"
        f"**Anomaly Score:** `{score:.6f}` | **Threshold:** `{THRESHOLD:.6f}`\n\n"
        f"### Explainable AI (XAI) Summary\n"
        f"{xai['summary']}"
    )

    xai_metrics = (
        f"Primary Location   : {xai['location']}\n"
        f"Morphological Type : {xai['pattern_type']}\n"
        f"Affected Area      : {xai['affected_area_percent']}% ({xai['region_count']} region(s))\n"
        f"Peak Confidence    : {xai['peak_confidence_percent']}% (ROI Mean: {xai['mean_roi_confidence_percent']}%)\n"
        f"Texture Attribution: {attr['shallow_texture_percent']}% (DINOv2 Layers 3–6)\n"
        f"Structure Attribution: {attr['deep_structure_percent']}% (DINOv2 Layers 9–12)"
    )

    return (
        result_text,
        xai_metrics,
        xai_data["heatmap_image"],
        xai_data["overlay_image"],
        xai_data["xai_annotated_image"],
    )


with gr.Blocks(title="AnomalyVFM+ XAI Demo") as demo:
    gr.Markdown("# AnomalyVFM+ — Explainable Zero-Shot Anomaly Detection")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            detect_button = gr.Button("Detect & Explain Anomaly", variant="primary")
        with gr.Column():
            result = gr.Markdown("Upload an image and click **Detect & Explain Anomaly**.")
            xai_box = gr.Textbox(label="XAI Diagnostic Metrics", lines=7, interactive=False)

    with gr.Row():
        heatmap_output = gr.Image(label="1. Raw Anomaly Heatmap")
        overlay_output = gr.Image(label="2. Blended Overlay")
        xai_output = gr.Image(label="3. XAI Contour & Bounding Box")

    detect_button.click(
        fn=predict,
        inputs=input_image,
        outputs=[result, xai_box, heatmap_output, overlay_output, xai_output],
    )


if __name__ == "__main__":
    demo.launch()