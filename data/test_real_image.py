import torch
from PIL import Image

from configs.loader import load_config
from data.mvtec import MVTecNormalDataset
from models.foundation import create_foundation_model
from torchvision import transforms


def main():
    print("=== Real MVTec → DINOv2 Test ===")

    config = load_config("configs/baseline.yaml")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MVTecNormalDataset(
        root="data/raw/mvtec",
        category="bottle",
        transform=transform
    )

    image_tensor = dataset[0]

    print("Dataset image shape:", image_tensor.shape)

    # Convert tensor [C,H,W] → PIL image
    image = image_tensor.permute(1, 2, 0)
    image = (image * 255).byte().numpy()
    image = Image.fromarray(image)

    # Create foundation model
    foundation_model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading DINOv2...")
    foundation_model.load()

    print("Extracting features...")

    with torch.no_grad():
        features = foundation_model.extract_features(image)

    print("DINOv2 feature shape:", features.shape)
    print("Real image feature test: SUCCESS")


if __name__ == "__main__":
    main()