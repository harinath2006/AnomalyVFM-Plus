import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


MODEL_NAME = "facebook/dinov2-base"


def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.eval()

    return processor, model


def extract_patch_features(image):
    processor, model = load_model()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove the CLS token.
    patch_tokens = outputs.last_hidden_state[:, 1:, :]

    # DINOv2-base with 224x224 input produces 256 patch tokens.
    # 256 = 16 x 16.
    batch_size, num_patches, feature_dim = patch_tokens.shape

    grid_size = int(num_patches ** 0.5)

    # Convert:
    # [B, 256, 768]
    #
    # into:
    # [B, 768, 16, 16]

    feature_map = patch_tokens.reshape(
        batch_size,
        grid_size,
        grid_size,
        feature_dim
    )

    feature_map = feature_map.permute(0, 3, 1, 2)

    return feature_map


if __name__ == "__main__":

    print("=== DINOv2 Spatial Feature Test ===")

    image = Image.new("RGB", (224, 224), "white")

    feature_map = extract_patch_features(image)

    print("Feature map shape:", feature_map.shape)

    print("Expected format:")
    print("[Batch, Channels, Height, Width]")

    print("DINOv2 spatial feature test: SUCCESS")