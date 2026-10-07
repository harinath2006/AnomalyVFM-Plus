import torch
from models.fusion.multiscale import (
    MultiScaleFeatureExtractor,
    MultiScaleFeatureFusion
)


def main():

    print("=== PHASE 8.1 FINAL VERIFICATION ===")

    # --------------------------------------------
    # Simulate DINOv2 hidden states
    # --------------------------------------------

    hidden_states = [
        torch.randn(2, 257, 768)
        for _ in range(13)
    ]

    # --------------------------------------------
    # Feature extraction
    # --------------------------------------------

    extractor = MultiScaleFeatureExtractor(
        selected_layers=(3, 6, 9, 12)
    )

    features = extractor(hidden_states)

    assert len(features) == 4

    print("1. Multi-scale extraction: PASS")

    for index, feature in enumerate(features):

        print(
            f"   Scale {index + 1}:",
            feature.shape
        )

        assert feature.shape == (
            2, 768, 16, 16
        )

    # --------------------------------------------
    # Fusion
    # --------------------------------------------

    fusion = MultiScaleFeatureFusion(
        in_channels=768,
        num_scales=4,
        out_channels=768
    )

    fused = fusion(features)

    assert fused.shape == (
        2, 768, 16, 16
    )

    print("2. Feature fusion: PASS")
    print("   Fused shape:", fused.shape)

    # --------------------------------------------
    # Gradient verification
    # --------------------------------------------

    loss = fused.mean()

    loss.backward()

    gradient_found = False

    for name, parameter in fusion.named_parameters():

        if parameter.grad is not None:

            gradient_found = True

            print(
                "3. Fusion gradient: PASS"
            )

            print(
                "   Parameter:",
                name
            )

            print(
                "   Mean gradient:",
                parameter.grad.abs().mean().item()
            )

            break

    assert gradient_found

    # --------------------------------------------
    # Final result
    # --------------------------------------------

    print()
    print("==========================================")
    print("PHASE 8.1 FINAL VERIFICATION: SUCCESS")
    print("==========================================")

    print()
    print("Multi-scale pipeline verified:")
    print("DINOv2 hidden states")
    print("→ Layer 3 / 6 / 9 / 12")
    print("→ spatial feature maps")
    print("→ concatenation")
    print("→ learnable 1×1 fusion")
    print("→ fused representation")


if __name__ == "__main__":
    main()