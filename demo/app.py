from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from inference.threshold import load_threshold


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = Path(
    "experiments/plus/checkpoint.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print("=" * 60)
print("AnomalyVFM+ DEMO")
print("=" * 60)

print(
    f"Device: {DEVICE}"
)

print(
    "Loading model..."
)

model = AnomalyVFMPlus()

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    state_dict = checkpoint[
        "model_state_dict"
    ]

elif "state_dict" in checkpoint:

    state_dict = checkpoint[
        "state_dict"
    ]

else:

    raise KeyError(
        "No compatible model state found. "
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )

load_result = model.load_state_dict(
    state_dict,
    strict=False
)

print(
    "Missing keys:",
    len(load_result.missing_keys)
)

print(
    "Unexpected keys:",
    len(load_result.unexpected_keys)
)

model = model.to(DEVICE)
model.eval()

THRESHOLD = load_threshold()

print(
    f"Calibrated threshold: {THRESHOLD:.6f}"
)

print(
    "Model loaded successfully."
)

print("=" * 60)


# ============================================================
# INFERENCE
# ============================================================

def predict(image):

    if image is None:

        return (
            "Please upload an image.",
            None,
            None,
            "No image provided."
        )

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    if isinstance(image, np.ndarray):

        image = Image.fromarray(
            image.astype(np.uint8)
        )

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        anomaly_map, image_logit = (
            model.forward_with_score(
                [image]
            )
        )

        score = torch.sigmoid(
            image_logit
        ).item()

        probability_map = torch.sigmoid(
            anomaly_map
        )

        # ----------------------------------------------------
        # Resize anomaly map to original image size
        # ----------------------------------------------------

        probability_map = F.interpolate(
            probability_map,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        heatmap = (
            probability_map[
                0,
                0
            ]
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if score >= THRESHOLD:

        prediction = "ANOMALOUS"

        reason = (
            "Image-level anomaly score exceeded "
            "the calibrated threshold."
        )

    else:

        prediction = "NORMAL"

        reason = (
            "Anomaly score is below the calibrated "
            "threshold."
        )

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap_uint8 = (
        heatmap * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    # Use matplotlib colormap without creating
    # another dependency on the model pipeline.
    import matplotlib

    colored_heatmap = (
        matplotlib.colormaps["jet"](
            heatmap
        )[
            :, :, :3
        ]
        * 255
    ).astype(
        np.uint8
    )

    heatmap_image = Image.fromarray(
        colored_heatmap
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        image,
        heatmap_image.resize(
            image.size
        ),
        alpha=0.45
    )

    # --------------------------------------------------------
    # Result text
    # --------------------------------------------------------

    result_text = (
        f"## {prediction}\n\n"
        f"**Anomaly Score:** `{score:.6f}`\n\n"
        f"**Threshold:** `{THRESHOLD:.6f}`\n\n"
        f"**Decision:** {reason}"
    )

    technical_info = (
        f"Model: AnomalyVFM+\n"
        f"Foundation model: DINOv2\n"
        f"Adapter: LoRA\n"
        f"Feature fusion: Multi-scale\n"
        f"Scoring: Image predictor\n"
        f"Device: {DEVICE}"
    )

    return (
        result_text,
        heatmap_image,
        overlay,
        technical_info
    )


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="AnomalyVFM+"
) as demo:

    gr.Markdown(
        """
        # 🔍 AnomalyVFM+

        ### Zero-Shot Visual Anomaly Detection

        Upload an image to detect whether it contains
        an anomaly using a Vision Foundation Model.
        """
    )

    gr.Markdown(
        """
        **Pipeline:** DINOv2 → LoRA → Multi-scale Feature Fusion
        → Anomaly Decoder + Image Predictor
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Product Image"
            )

            detect_button = gr.Button(
                "🔍 Detect Anomaly",
                variant="primary"
            )

        with gr.Column():

            result = gr.Markdown(
                value="Upload an image and click **Detect Anomaly**."
            )

            technical_info = gr.Textbox(
                label="Technical Information",
                lines=6,
                interactive=False
            )

    with gr.Row():

        heatmap_output = gr.Image(
            label="Anomaly Heatmap"
        )

        overlay_output = gr.Image(
            label="Anomaly Overlay"
        )

    gr.Markdown(
        """
        ---
        
        ### How it works

        1. The uploaded image is processed by **DINOv2**.
        2. **LoRA adaptation** extracts task-specific features.
        3. Features from multiple transformer layers are fused.
        4. The anomaly decoder produces a spatial anomaly map.
        5. The image predictor produces the final anomaly score.
        6. A calibrated threshold determines **NORMAL** or **ANOMALOUS**.
        """
    )

    detect_button.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            result,
            heatmap_output,
            overlay_output,
            technical_info
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch()