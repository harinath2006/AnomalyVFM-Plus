import base64
import io
import os
import urllib.request
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.xai.explainer import explain_prediction


APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path("/tmp/anomalyvfm")
CHECKPOINT_PATH = MODEL_DIR / "checkpoint.pt"

# Also check local checkpoint first when running locally via uvicorn
LOCAL_CHECKPOINT_PATH = APP_DIR / "experiments" / "plus" / "checkpoint.pt"

CHECKPOINT_URL = os.getenv(
    "ANOMALYVFM_CHECKPOINT_URL",
    "https://github.com/harinath2006/AnomalyVFM-Plus/releases/download/v2.0.0/checkpoint.pt",
)

DEVICE = torch.device("cpu")
MAX_OUTPUT_SIZE = 512
DEFAULT_THRESHOLD = 0.311792

app = FastAPI(
    title="AnomalyVFM+ API",
    description="Zero-Shot Visual Anomaly Detection with Explainable AI",
    version="2.0.0",
)

_model = None
_threshold = None


def ensure_checkpoint():
    if LOCAL_CHECKPOINT_PATH.exists():
        return LOCAL_CHECKPOINT_PATH
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("checkpoint.pt was not found after download.")
    return CHECKPOINT_PATH


def load_decision_threshold():
    threshold_value = os.getenv("ANOMALY_THRESHOLD")
    if threshold_value is not None:
        try:
            return float(threshold_value)
        except ValueError:
            pass
    return DEFAULT_THRESHOLD


def load_model():
    global _model, _threshold
    if _model is not None:
        return _model, _threshold

    checkpoint_path = ensure_checkpoint()
    model = AnomalyVFMPlus()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    cleaned_state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    _model = model
    _threshold = load_decision_threshold()
    return _model, _threshold


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
def root():
    frontend_path = APP_DIR / "public" / "index.html"
    if not frontend_path.exists():
        return {"name": "AnomalyVFM+", "status": "online", "endpoint": "/api/predict"}
    return FileResponse(frontend_path)


@app.get("/health")
def health():
    return {"status": "online", "model": "AnomalyVFM+"}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        model, threshold = load_model()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {error}")

    try:
        with torch.no_grad():
            anomaly_map, image_logit = model.forward_with_score([image])
            image_score = torch.sigmoid(image_logit).reshape(-1)[0].item()

        original_width, original_height = image.size
        scale = min(MAX_OUTPUT_SIZE / original_width, MAX_OUTPUT_SIZE / original_height, 1.0)
        output_width = max(1, int(original_width * scale))
        output_height = max(1, int(original_height * scale))

        xai_data = explain_prediction(
            model=model,
            pil_image=image,
            anomaly_map=anomaly_map,
            image_score=image_score,
            threshold=threshold,
            output_size=(output_width, output_height),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference failed: {error}")

    if image_score >= threshold:
        prediction = "ANOMALOUS"
        reason = "The model anomaly score is above the calibrated decision threshold."
    else:
        prediction = "NORMAL"
        reason = "The model anomaly score is below the calibrated decision threshold."

    return {
        "prediction": prediction,
        "anomaly_score": round(image_score, 6),
        "threshold": round(threshold, 6),
        "reason": reason,
        "heatmap": image_to_base64(xai_data["heatmap_image"]),
        "overlay": image_to_base64(xai_data["overlay_image"]),
        "xai_overlay": image_to_base64(xai_data["xai_annotated_image"]),
        "xai_report": xai_data["xai_report"],
        "technical_info": {
            "model": "AnomalyVFM+",
            "foundation_model": "DINOv2",
            "adaptation": "LoRA (Rank 8)",
            "feature_fusion": "Multi-scale (Layers 3, 6, 9, 12)",
            "decoder": "Progressive Upsampling Decoder",
            "xai_engine": "Multi-Scale Attribution + Spatial ROI Localization",
            "device": "CPU",
            "output_size": (output_width, output_height),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.predict:app", host="127.0.0.1", port=8000, reload=True)