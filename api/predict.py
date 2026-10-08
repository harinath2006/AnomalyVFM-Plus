import base64
import io
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = Path("/tmp/anomalyvfm")
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.pt"

CHECKPOINT_URL = os.getenv(
    "ANOMALYVFM_CHECKPOINT_URL",
    "https://github.com/harinath2006/AnomalyVFM-Plus/releases/download/v1.0.0/checkpoint.pt"
)

DEVICE = torch.device("cpu")

MAX_OUTPUT_SIZE = 512

# Calibrated using MVTec Bottle validation split.
# Can be overridden on Vercel with ANOMALY_THRESHOLD.
DEFAULT_THRESHOLD = 0.081440


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AnomalyVFM+ API",
    description="Zero-Shot Visual Anomaly Detection using Vision Foundation Models",
    version="1.0.0"
)


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

_model = None
_threshold = None


# ============================================================
# CHECKPOINT DOWNLOAD
# ============================================================

def ensure_checkpoint():
    """
    Download the trained .pt checkpoint directly if it does not
    already exist in the temporary Vercel filesystem.
    """

    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint already exists: {CHECKPOINT_PATH}")
        return CHECKPOINT_PATH

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading AnomalyVFM+ checkpoint...")
    print(f"Source: {CHECKPOINT_URL}")

    try:
        urllib.request.urlretrieve(
            CHECKPOINT_URL,
            CHECKPOINT_PATH
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to download checkpoint: {error}"
        )

    print("Checkpoint downloaded.")
    print(f"Checkpoint ready: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "checkpoint.pt was not found after download."
        )

    return CHECKPOINT_PATH


# ============================================================
# THRESHOLD
# ============================================================

def load_decision_threshold():
    """
    Load the calibrated anomaly decision threshold.

    Vercel:
        Uses ANOMALY_THRESHOLD environment variable if provided.

    Local/default:
        Uses the calibrated threshold from the MVTec Bottle experiment.
    """

    threshold_value = os.getenv("ANOMALY_THRESHOLD")

    if threshold_value is not None:
        try:
            threshold = float(threshold_value)
            print(
                f"Using threshold from environment: {threshold:.6f}"
            )
            return threshold
        except ValueError:
            print(
                "Invalid ANOMALY_THRESHOLD environment variable. "
                "Using default threshold."
            )

    print(
        f"Using calibrated default threshold: "
        f"{DEFAULT_THRESHOLD:.6f}"
    )

    return DEFAULT_THRESHOLD


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    global _model
    global _threshold

    # Reuse model between warm Vercel invocations.
    if _model is not None:
        return _model, _threshold

    # --------------------------------------------------------
    # Download checkpoint
    # --------------------------------------------------------

    checkpoint_path = ensure_checkpoint()

    print("Loading AnomalyVFM+ model...")

    # --------------------------------------------------------
    # Create model architecture
    # --------------------------------------------------------

    model = AnomalyVFMPlus()

    # --------------------------------------------------------
    # Load trained checkpoint
    # --------------------------------------------------------

    print("Loading checkpoint weights...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint:
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Restore trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # Load calibrated threshold
    # --------------------------------------------------------

    threshold = load_decision_threshold()

    # Cache model for warm invocations.
    _model = model
    _threshold = threshold

    print("AnomalyVFM+ model loaded successfully.")
    print(
        f"Decision threshold: {threshold:.6f}"
    )

    return _model, _threshold


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(image: Image.Image) -> str:

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


# ============================================================
# ROOT / FRONTEND
# ============================================================

@app.get("/")
def root():

    frontend_path = (
        APP_DIR
        / "public"
        / "index.html"
    )

    if not frontend_path.exists():

        return {
            "name": "AnomalyVFM+",
            "status": "online",
            "model": "DINOv2 + LoRA + Multi-scale Fusion",
            "endpoint": "/api/predict"
        }

    return FileResponse(
        frontend_path
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "model": "AnomalyVFM+"
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file type
    # --------------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="File type could not be determined."
        )

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image file."
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model, threshold = load_model()

    except Exception as error:

        print(f"Model loading error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {error}"
        )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            anomaly_map, image_logit = (
                model.forward_with_score(
                    [image]
                )
            )

            # Image-level anomaly probability.
            image_score = torch.sigmoid(
                image_logit
            )[0].item()

            # Pixel-level anomaly probability.
            probability_map = torch.sigmoid(
                anomaly_map
            )

    except Exception as error:

        print(f"Inference error: {error}")

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {error}"
        )

    # ========================================================
    # CREATE COMPACT VISUALIZATION
    # ========================================================

    original_width, original_height = image.size

    # --------------------------------------------------------
    # Resize visualization to maximum 512x512
    # --------------------------------------------------------

    scale = min(
        MAX_OUTPUT_SIZE / original_width,
        MAX_OUTPUT_SIZE / original_height,
        1.0
    )

    output_width = max(
        1,
        int(original_width * scale)
    )

    output_height = max(
        1,
        int(original_height * scale)
    )

    # --------------------------------------------------------
    # Resize anomaly map
    # --------------------------------------------------------

    probability_map = F.interpolate(
        probability_map,
        size=(
            output_height,
            output_width
        ),
        mode="bilinear",
        align_corners=False
    )

    heatmap = (
        probability_map[0, 0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    import numpy as np
    import matplotlib

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0
    )

    colored_heatmap = (
        matplotlib.colormaps["jet"](
            heatmap
        )[:, :, :3] * 255
    ).astype(np.uint8)

    heatmap_image = Image.fromarray(
        colored_heatmap
    )

    # --------------------------------------------------------
    # Resize original image
    # --------------------------------------------------------

    display_image = image.resize(
        (
            output_width,
            output_height
        ),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = Image.blend(
        display_image,
        heatmap_image,
        alpha=0.45
    )

    # ========================================================
    # FINAL DECISION
    # ========================================================

    if image_score >= threshold:

        prediction = "ANOMALOUS"

        reason = (
            "The model anomaly score is above the "
            "calibrated decision threshold."
        )

    else:

        prediction = "NORMAL"

        reason = (
            "The model anomaly score is below the "
            "calibrated decision threshold."
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "prediction": prediction,

        "anomaly_score": round(
            image_score,
            6
        ),

        "threshold": round(
            threshold,
            6
        ),

        "reason": reason,

        "heatmap": image_to_base64(
            heatmap_image
        ),

        "overlay": image_to_base64(
            overlay
        ),

        "technical_info": {

            "model": "AnomalyVFM+",

            "foundation_model": "DINOv2",

            "adaptation": "LoRA",

            "feature_fusion": "Multi-scale",

            "decoder": "Anomaly Decoder",

            "device": "CPU",

            "output_size": (
                output_width,
                output_height
            )
        }
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.predict:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )