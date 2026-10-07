import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from configs.loader import load_config
from data.mvtec import MVTecNormalDataset
from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder
from synthetic.masks import create_rectangle_mask
from synthetic.generate import apply_synthetic_anomaly
from training.losses import anomaly_loss


def train_decoder(max_batches=2):
    print("=== Real MVTec Training Test ===")

    config = load_config("configs/baseline.yaml")

    # --------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MVTecNormalDataset(
        root="data/raw/mvtec",
        category="bottle",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    print("Dataset size:", len(dataset))

    # --------------------------------------------------
    # 2. Foundation model
    # --------------------------------------------------
    foundation_model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading foundation model...")
    foundation_model.load()

    foundation_model.model.eval()

    # --------------------------------------------------
    # 3. Decoder
    # --------------------------------------------------
    decoder_config = config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(
            decoder_config["hidden_channels"]
        )
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=1e-4
    )

    decoder.train()

    # --------------------------------------------------
    # 4. Training
    # --------------------------------------------------
    for batch_index, images in enumerate(loader):

        if batch_index >= max_batches:
            break

        print(
            f"\nBatch {batch_index + 1}/{max_batches}"
        )

        batch_size = images.shape[0]

        # --------------------------------------------------
        # Create synthetic anomaly masks
        # --------------------------------------------------
        masks = create_rectangle_mask(
            batch_size=batch_size,
            height=224,
            width=224,
            top=80,
            left=60,
            rectangle_height=40,
            rectangle_width=70
        )

        anomalous_images = apply_synthetic_anomaly(
            images,
            masks
        )

        # --------------------------------------------------
        # Convert tensors → PIL images
        # --------------------------------------------------
        pil_images = []

        for image in anomalous_images:
            image = image.permute(1, 2, 0)
            image = (image * 255).byte().numpy()
            image = Image.fromarray(image)
            pil_images.append(image)

        # --------------------------------------------------
        # DINOv2 feature extraction
        # --------------------------------------------------
        with torch.no_grad():
            features = foundation_model.extract_features(
                pil_images
            )

        print(
            "Feature shape:",
            features.shape
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------
        logits = decoder(features)

        logits = F.interpolate(
            logits,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------
        loss = anomaly_loss(
            logits,
            masks
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            f"Loss: {loss.item():.4f}"
        )

    print("\nReal MVTec training test: SUCCESS")


if __name__ == "__main__":
    train_decoder()