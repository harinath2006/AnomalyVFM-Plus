import torch
from torchvision import transforms

from configs.loader import load_config
from data.mvtec import MVTecNormalDataset
from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder


def main():
    print("=== Real MVTec → DINOv2 → Decoder Test ===")

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

    print("Input image shape:", image_tensor.shape)

    # Convert tensor to PIL image for DINOv2
    image = image_tensor.permute(1, 2, 0)
    image = (image * 255).byte().numpy()

    from PIL import Image
    image = Image.fromarray(image)

    # Foundation model
    foundation_model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading DINOv2...")
    foundation_model.load()

    print("Extracting features...")

    with torch.no_grad():
        features = foundation_model.extract_features(image)

    print("Feature shape:", features.shape)

    # Decoder
    decoder_config = config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(
            decoder_config["hidden_channels"]
        )
    )

    decoder.eval()

    print("Generating anomaly prediction...")

    anomaly_map, score = decoder.predict(
        features,
        output_size=decoder_config["output_size"]
    )

    print("Anomaly map shape:", anomaly_map.shape)
    print("Anomaly score:", score)
    print("Real MVTec decoder test: SUCCESS")


if __name__ == "__main__":
    main()