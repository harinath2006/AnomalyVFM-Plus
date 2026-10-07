import torch
import torch.nn.functional as F
from torchvision import transforms

from configs.loader import load_config
from data.mvtec import MVTecNormalDataset
from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder
from models.adapters.inject_lora import inject_lora
from synthetic.masks import create_rectangle_mask
from synthetic.generate import apply_synthetic_anomaly
from training.losses import anomaly_loss


def main():

    print("=== LoRA + Decoder Training Step Test ===")

    config = load_config("configs/baseline.yaml")

    # --------------------------------------------------
    # 1. Load one real MVTec image
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

    image = dataset[0]

    print("Original image shape:", image.shape)

    # Add batch dimension
    image = image.unsqueeze(0)

    # --------------------------------------------------
    # 2. Create synthetic anomaly
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

    anomalous_image = apply_synthetic_anomaly(
        image,
        mask
    )

    print("Anomalous image shape:", anomalous_image.shape)
    print("Mask shape:", mask.shape)

    # --------------------------------------------------
    # 3. Convert tensor to PIL
    # --------------------------------------------------

    to_pil = transforms.ToPILImage()

    pil_image = to_pil(anomalous_image[0])

    # --------------------------------------------------
    # 4. Load DINOv2
    # --------------------------------------------------

    foundation = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading DINOv2...")
    foundation.load()

    # --------------------------------------------------
    # 5. Inject LoRA
    # --------------------------------------------------

    print("Injecting LoRA...")

    lora_count = inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    print("LoRA layers:", lora_count)

    # --------------------------------------------------
    # 6. Create decoder
    # --------------------------------------------------

    decoder_config = config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(
            decoder_config["hidden_channels"]
        )
    )

    # --------------------------------------------------
    # 7. Optimizer
    # --------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in foundation.model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameters += list(
        decoder.parameters()
    )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=1e-4
    )

    # --------------------------------------------------
    # 8. Forward pass
    # --------------------------------------------------

    features = foundation.extract_features(
        pil_image,
        trainable=True
    )

    print("Feature shape:", features.shape)
    print("Features require grad:", features.requires_grad)

    logits = decoder(features)

    logits = F.interpolate(
        logits,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    print("Decoder output:", logits.shape)

    # --------------------------------------------------
    # 9. Calculate anomaly loss
    # --------------------------------------------------

    loss = anomaly_loss(
        logits,
        mask
    )

    print("Loss:", loss.item())

    # --------------------------------------------------
    # 10. Backpropagation
    # --------------------------------------------------

    optimizer.zero_grad()

    loss.backward()

    # --------------------------------------------------
    # 11. Check LoRA gradient
    # --------------------------------------------------

    lora_gradient_found = False

    for name, parameter in foundation.model.named_parameters():

        if (
            "lora_B" in name
            and parameter.grad is not None
        ):

            gradient = parameter.grad.abs().mean().item()

            print(
                "LoRA gradient:",
                name,
                "->",
                gradient
            )

            if gradient > 0:
                lora_gradient_found = True

            break

    # --------------------------------------------------
    # 12. Check decoder gradient
    # --------------------------------------------------

    decoder_gradient_found = False

    for name, parameter in decoder.named_parameters():

        if parameter.grad is not None:

            gradient = parameter.grad.abs().mean().item()

            print(
                "Decoder gradient:",
                name,
                "->",
                gradient
            )

            if gradient > 0:
                decoder_gradient_found = True

            break

    # --------------------------------------------------
    # 13. Update parameters
    # --------------------------------------------------

    optimizer.step()

    print("\nOptimizer step completed.")

    if lora_gradient_found and decoder_gradient_found:
        print("\n=== LoRA TRAINING STEP: SUCCESS ===")
    else:
        print("\n=== LoRA TRAINING STEP: FAILED ===")


if __name__ == "__main__":
    main()