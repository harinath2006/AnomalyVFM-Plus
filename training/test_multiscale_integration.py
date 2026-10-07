import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.mvtec import MVTecNormalDataset

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.fusion.multiscale import MultiScaleFeatureFusion

from synthetic.irregular import create_irregular_mask
from synthetic.corruptions import apply_anomaly_corruption
from synthetic.feature_difference import compute_feature_difference

from training.confidence_loss import confidence_weighted_loss


def get_multiscale_features(
    foundation,
    images,
    trainable=True
):
    """
    Extract selected DINOv2 hidden states and convert
    them into spatial feature maps.

    Returns:
        [Layer 3, Layer 6, Layer 9, Layer 12]
    """

    inputs = foundation.processor(
        images=images,
        return_tensors="pt"
    )

    if trainable:
        outputs = foundation.model(
            **inputs,
            output_hidden_states=True
        )
    else:
        with torch.no_grad():
            outputs = foundation.model(
                **inputs,
                output_hidden_states=True
            )

    selected_layers = (3, 6, 9, 12)

    features = []

    for layer_index in selected_layers:

        tokens = outputs.hidden_states[layer_index]

        # Remove CLS token.
        patch_tokens = tokens[:, 1:, :]

        batch_size = patch_tokens.shape[0]
        num_patches = patch_tokens.shape[1]
        channels = patch_tokens.shape[2]

        grid_size = int(num_patches ** 0.5)

        feature_map = patch_tokens.reshape(
            batch_size,
            grid_size,
            grid_size,
            channels
        )

        feature_map = feature_map.permute(
            0,
            3,
            1,
            2
        )

        features.append(feature_map)

    return features


def main():

    print("=== Multi-Scale Integration Test ===")

    device = torch.device("cpu")

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

    images = next(iter(loader)).to(device)

    print("Dataset images:", images.shape)

    # --------------------------------------------------
    # DINOv2
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
    # Multi-scale fusion
    # --------------------------------------------------

    fusion = MultiScaleFeatureFusion(
        in_channels=768,
        num_scales=4,
        out_channels=768
    ).to(device)

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------

    decoder = AnomalyDecoder(
        in_channels=768
    ).to(device)

    # --------------------------------------------------
    # Synthetic anomaly
    # --------------------------------------------------

    mask = create_irregular_mask(
        batch_size=images.shape[0],
        height=224,
        width=224
    ).to(device)

    anomalous_images = apply_anomaly_corruption(
        images,
        mask,
        corruption_type="mixed"
    )

    # --------------------------------------------------
    # Confidence map
    # --------------------------------------------------

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

        _, confidence_map = compute_feature_difference(
            normal_features,
            anomaly_features_for_confidence,
            output_size=(224, 224)
        )

    # --------------------------------------------------
    # Trainable multi-scale features
    # --------------------------------------------------

    foundation.model.train()
    fusion.train()
    decoder.train()

    multiscale_features = get_multiscale_features(
        foundation,
        anomalous_images,
        trainable=True
    )

    print(
        "Number of scales:",
        len(multiscale_features)
    )

    for index, feature in enumerate(
        multiscale_features
    ):
        print(
            f"Scale {index + 1}:",
            feature.shape
        )

    # --------------------------------------------------
    # Fusion
    # --------------------------------------------------

    fused_features = fusion(
        multiscale_features
    )

    print(
        "Fused feature shape:",
        fused_features.shape
    )

    print(
        "Fused features require grad:",
        fused_features.requires_grad
    )

    # --------------------------------------------------
    # Decoder
    # --------------------------------------------------

    predictions = decoder(
        fused_features
    )

    print(
        "Decoder output:",
        predictions.shape
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss = confidence_weighted_loss(
        predictions,
        mask,
        confidence_map
    )

    print(
        "Loss:",
        loss.item()
    )

    # --------------------------------------------------
    # Backpropagation
    # --------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in foundation.model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameters += list(
        fusion.parameters()
    )

    trainable_parameters += list(
        decoder.parameters()
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=1e-4
    )

    optimizer.zero_grad()

    loss.backward()

    # --------------------------------------------------
    # Verify LoRA gradient
    # --------------------------------------------------

    lora_gradient_found = False

    for name, parameter in foundation.model.named_parameters():

        if (
            "lora_B" in name
            and parameter.grad is not None
        ):

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
    # Verify fusion gradient
    # --------------------------------------------------

    fusion_gradient_found = False

    for name, parameter in fusion.named_parameters():

        if parameter.grad is not None:

            print(
                "Fusion gradient:",
                name,
                "->",
                parameter.grad.abs().mean().item()
            )

            fusion_gradient_found = True
            break

    print(
        "Fusion gradient received:",
        fusion_gradient_found
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer.step()

    print("Optimizer step completed.")

    # --------------------------------------------------
    # Verification
    # --------------------------------------------------

    assert fused_features.shape == (
        2, 768, 16, 16
    )

    assert fused_features.requires_grad

    assert lora_gradient_found

    assert fusion_gradient_found

    print()
    print(
        "=== Multi-Scale Integration Test: SUCCESS ==="
    )


if __name__ == "__main__":
    main()