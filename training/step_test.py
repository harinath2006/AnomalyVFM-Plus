import torch
import torch.nn.functional as F
from PIL import Image

from configs.loader import load_config
from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder
from synthetic.masks import create_rectangle_mask
from synthetic.generate import apply_synthetic_anomaly
from training.losses import anomaly_loss


def main():
    print("=== Single Training Step Test ===")

    config = load_config("configs/baseline.yaml")

    # --------------------------------------------------
    # 1. Create a simple normal image
    # --------------------------------------------------
    image = torch.rand(1, 3, 224, 224)

    # --------------------------------------------------
    # 2. Create synthetic anomaly mask
    # --------------------------------------------------
    mask = create_rectangle_mask(
        batch_size=1,
        height=224,
        width=224,
        top=80,
        left=60,
        rectangle_height=40,
        rectangle_width=70
    )

    # --------------------------------------------------
    # 3. Apply synthetic anomaly
    # --------------------------------------------------
    anomalous_image = apply_synthetic_anomaly(
        image,
        mask
    )

    # Convert tensor to PIL for DINOv2
    image_for_model = anomalous_image[0].permute(1, 2, 0)
    image_for_model = (image_for_model * 255).byte().numpy()
    image_for_model = Image.fromarray(image_for_model)

    # --------------------------------------------------
    # 4. Load frozen foundation model
    # --------------------------------------------------
    foundation_model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading foundation model...")
    foundation_model.load()

    print("Extracting features...")

    with torch.no_grad():
        features = foundation_model.extract_features(
            image_for_model
        )

    print("Feature shape:", features.shape)

    # --------------------------------------------------
    # 5. Create decoder
    # --------------------------------------------------
    decoder_config = config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(
            decoder_config["hidden_channels"]
        )
    )

    # --------------------------------------------------
    # 6. Forward pass
    # --------------------------------------------------
    logits = decoder(features)

    logits = F.interpolate(
        logits,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    print("Prediction shape:", logits.shape)

    # --------------------------------------------------
    # 7. Calculate loss
    # --------------------------------------------------
    loss = anomaly_loss(
        logits,
        mask
    )

    print("Loss:", loss.item())

    # --------------------------------------------------
    # 8. Backpropagation
    # --------------------------------------------------
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=1e-4
    )

    optimizer.zero_grad()

    loss.backward()

    # Check whether decoder received gradients
    gradient_found = False

    for parameter in decoder.parameters():
        if parameter.grad is not None:
            gradient_found = True
            break

    print("Decoder gradients received:", gradient_found)

    optimizer.step()

    print("Training step test: SUCCESS")


if __name__ == "__main__":
    main()