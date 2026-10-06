import torch

from configs.loader import load_config
from models.foundation import create_foundation_model


def main():
    config = load_config("configs/baseline.yaml")

    model_name = config["model"]["name"]
    image_size = config["input"]["image_size"]

    print("=== Foundation Model Runner ===")
    print("Model:", model_name)
    print("Image size:", image_size)

    model = create_foundation_model(model_name)

    print("Loading foundation model...")
    model.load()

    # Dummy image tensor for pipeline testing
    image = torch.rand(1, 3, image_size, image_size)

    print("Extracting features...")
    features = model.extract_features(image)

    print("Feature shape:", features.shape)
    print("Foundation pipeline: SUCCESS")


if __name__ == "__main__":
    main()