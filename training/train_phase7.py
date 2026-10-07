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

    print("=== Phase 7 Integrated Training Test ===")

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device("cpu")
    print("Device:", device)

    # --------------------------------------------------
    # Dataset
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
    # Foundation model
    # --------------------------------------------------

    foundation = DINOv2()
    foundation.load()

    foundation.model.to(device)

    # --------------------------------------------------
    # LoRA
    # --------------------------------------------------

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

    print("LoRA layers:", injected)

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------

    decoder = AnomalyDecoder(
    in_channels=768
    ).to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in foundation.model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameters += list(
        decoder.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-4
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    foundation.model.train()
    decoder.train()

    for step, images in enumerate(loader):

        if step >= 2:
            break

        images = images.to(device)

        print()
        print("----- Batch", step + 1, "-----")

        # --------------------------------------------------
        # Create irregular anomaly
        # --------------------------------------------------

        mask = create_irregular_mask(
            batch_size=images.shape[0],
            height=224,
            width=224
        ).to(device)

        # --------------------------------------------------
        # Apply realistic corruption
        # --------------------------------------------------

        anomalous_images = apply_anomaly_corruption(
            images,
            mask,
            corruption_type="mixed"
        )

        # --------------------------------------------------
        # Feature difference for confidence
        #
        # Confidence is treated as a training target.
        # We do not backpropagate through this calculation.
        # --------------------------------------------------

        with torch.no_grad():

            normal_features_for_confidence = (
                foundation.extract_features(
                    images,
                    trainable=False
                )
            )

            anomaly_features_for_confidence = (
                foundation.extract_features(
                    anomalous_images,
                    trainable=False
                )
            )

            _, confidence_map = compute_feature_difference(
                normal_features_for_confidence,
                anomaly_features_for_confidence,
                output_size=(224, 224)
            )

        # --------------------------------------------------
        # Trainable anomaly feature extraction
        # --------------------------------------------------

        anomaly_features = foundation.extract_features(
            anomalous_images,
            trainable=True
        )

        print(
            "Feature shape:",
            anomaly_features.shape
        )

        print(
            "Features require grad:",
            anomaly_features.requires_grad
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        predictions = decoder(
            anomaly_features
        )

        print(
            "Prediction shape:",
            predictions.shape
        )

        # --------------------------------------------------
        # Confidence-weighted loss
        # --------------------------------------------------

        loss = confidence_weighted_loss(
            predictions,
            mask,
            confidence_map
        )

        print(
            "Confidence weighted loss:",
            loss.item()
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # --------------------------------------------------
        # Check LoRA gradient
        # --------------------------------------------------

        lora_gradient_found = False

        for name, parameter in foundation.model.named_parameters():

            if "lora_B" in name and parameter.grad is not None:

                print(
                    "LoRA gradient:",
                    name,
                    "->",
                    parameter.grad.abs().mean().item()
                )

                lora_gradient_found = True
                break

        print(
            "LoRA gradient received:",
            lora_gradient_found
        )

        # --------------------------------------------------
        # Optimizer step
        # --------------------------------------------------

        optimizer.step()

        print(
            "Optimizer step completed."
        )

    print()
    print("=== Phase 7 Integrated Training Test: SUCCESS ===")


if __name__ == "__main__":
    main()