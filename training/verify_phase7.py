import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.mvtec import MVTecNormalDataset

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.decoder.anomaly_decoder import AnomalyDecoder

from synthetic.irregular import create_irregular_mask
from synthetic.corruptions import apply_anomaly_corruption
from synthetic.feature_difference import compute_feature_difference

from training.confidence_loss import confidence_weighted_loss


def main():

    print("=== PHASE 7 FINAL VERIFICATION ===")

    device = torch.device("cpu")

    # --------------------------------------------
    # Dataset
    # --------------------------------------------

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
        shuffle=False
    )

    images = next(iter(loader)).to(device)

    print("1. Dataset: PASS")
    print("   Image shape:", images.shape)

    # --------------------------------------------
    # Foundation model
    # --------------------------------------------

    foundation = DINOv2()
    foundation.load()
    foundation.model.to(device)

    # --------------------------------------------
    # LoRA
    # --------------------------------------------

    injected = inject_lora(
        foundation.model,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ),
        rank=4,
        alpha=8
    )

    assert injected == 48

    print("2. LoRA injection: PASS")
    print("   LoRA layers:", injected)

    # --------------------------------------------
    # Synthetic anomaly mask
    # --------------------------------------------

    mask = create_irregular_mask(
        batch_size=images.shape[0],
        height=224,
        width=224
    )

    assert mask.shape == (2, 1, 224, 224)
    assert mask.min() >= 0
    assert mask.max() <= 1

    print("3. Irregular mask: PASS")

    # --------------------------------------------
    # Synthetic corruption
    # --------------------------------------------

    anomalous_images = apply_anomaly_corruption(
        images,
        mask,
        corruption_type="mixed"
    )

    assert anomalous_images.shape == images.shape
    assert anomalous_images.min() >= 0
    assert anomalous_images.max() <= 1

    print("4. Synthetic corruption: PASS")

    # --------------------------------------------
    # Feature extraction
    # --------------------------------------------

    with torch.no_grad():

        normal_features = foundation.extract_features(
            images,
            trainable=False
        )

        anomaly_features_for_confidence = (
            foundation.extract_features(
                anomalous_images,
                trainable=False
            )
        )

    assert normal_features.shape == (
        2, 768, 16, 16
    )

    print("5. Feature extraction: PASS")
    print("   Feature shape:", normal_features.shape)

    # --------------------------------------------
    # Feature difference
    # --------------------------------------------

    _, confidence_map = compute_feature_difference(
        normal_features,
        anomaly_features_for_confidence,
        output_size=(224, 224)
    )

    assert confidence_map.shape == (
        2, 1, 224, 224
    )

    assert confidence_map.min() >= 0
    assert confidence_map.max() <= 1

    print("6. Confidence map: PASS")

    # --------------------------------------------
    # Trainable anomaly features
    # --------------------------------------------

    foundation.model.train()

    anomaly_features = foundation.extract_features(
        anomalous_images,
        trainable=True
    )

    assert anomaly_features.requires_grad

    print("7. Trainable LoRA features: PASS")

    # --------------------------------------------
    # Decoder
    # --------------------------------------------

    decoder = AnomalyDecoder(
        in_channels=768
    ).to(device)

    decoder.train()

    predictions = decoder(
        anomaly_features
    )

    assert predictions.shape[0] == 2
    assert predictions.shape[1] == 1

    print("8. Decoder: PASS")
    print("   Prediction shape:", predictions.shape)

    # --------------------------------------------
    # Confidence-weighted loss
    # --------------------------------------------

    loss = confidence_weighted_loss(
        predictions,
        mask,
        confidence_map
    )

    assert torch.isfinite(loss)
    assert loss.item() > 0

    print("9. Confidence-weighted loss: PASS")
    print("   Loss:", loss.item())

    # --------------------------------------------
    # Backpropagation
    # --------------------------------------------

    loss.backward()

    gradient_found = False

    for name, parameter in foundation.model.named_parameters():

        if "lora_B" in name and parameter.grad is not None:

            gradient_found = True

            print(
                "10. LoRA gradient: PASS"
            )

            print(
                "    Parameter:",
                name
            )

            print(
                "    Mean gradient:",
                parameter.grad.abs().mean().item()
            )

            break

    assert gradient_found

    # --------------------------------------------
    # Final result
    # --------------------------------------------

    print()
    print("==========================================")
    print("PHASE 7 FINAL VERIFICATION: SUCCESS")
    print("==========================================")

    print()
    print("Phase 7 pipeline verified:")
    print("MVTec → irregular mask → corruption")
    print("→ DINOv2 → feature difference")
    print("→ confidence map → decoder")
    print("→ confidence-weighted loss → LoRA gradient")


if __name__ == "__main__":
    main()