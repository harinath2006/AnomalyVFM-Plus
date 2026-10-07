import torch
from PIL import Image

from models.foundation import create_foundation_model
from models.adapters.inject_lora import inject_lora
from models.adapters.lora import LoRALinear


def main():
    print("=== LoRA Forward Pass Test ===")

    # Create image
    image = Image.new("RGB", (224, 224), "white")

    # Load original DINOv2
    foundation = create_foundation_model("dinov2")

    print("Loading DINOv2...")
    foundation.load()

    # Get original features
    with torch.no_grad():
        original_features = foundation.extract_features(image)

    print("Original feature shape:", original_features.shape)

    # Inject LoRA
    inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    # Get features after LoRA injection
    with torch.no_grad():
        lora_features = foundation.extract_features(image)

    print("LoRA feature shape:", lora_features.shape)

    # Compare
    difference = torch.abs(
        original_features - lora_features
    ).mean().item()

    print("Difference before LoRA training:", difference)

    print("\nLoRA forward test: SUCCESS")


if __name__ == "__main__":
    main()