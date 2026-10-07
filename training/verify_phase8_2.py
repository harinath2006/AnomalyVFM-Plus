import torch

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== PHASE 8.2 FINAL VERIFICATION ===")

    # --------------------------------------------------
    # Model creation
    # --------------------------------------------------

    model = AnomalyVFMPlus()

    print("1. AnomalyVFM+ model: PASS")
    print(
        "   LoRA layers:",
        model.lora_layers
    )

    assert model.lora_layers == 48

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    images = torch.rand(
        2,
        3,
        224,
        224
    )

    predictions = model(images)

    print("2. Forward pass: PASS")
    print(
        "   Input:",
        images.shape
    )
    print(
        "   Prediction:",
        predictions.shape
    )

    assert predictions.shape == (
        2,
        1,
        16,
        16
    )

    # --------------------------------------------------
    # Gradient flow
    # --------------------------------------------------

    loss = predictions.mean()

    loss.backward()

    # LoRA gradient
    lora_gradient = False

    for name, parameter in (
        model.foundation_model.named_parameters()
    ):

        if (
            "lora_B" in name
            and parameter.grad is not None
        ):

            lora_gradient = True
            break

    print(
        "3. LoRA gradient: ",
        "PASS" if lora_gradient else "FAIL"
    )

    assert lora_gradient

    # Fusion gradient
    fusion_gradient = False

    for parameter in model.fusion.parameters():

        if parameter.grad is not None:

            fusion_gradient = True
            break

    print(
        "4. Fusion gradient: ",
        "PASS" if fusion_gradient else "FAIL"
    )

    assert fusion_gradient

    # Decoder gradient
    decoder_gradient = False

    for parameter in model.decoder.parameters():

        if parameter.grad is not None:

            decoder_gradient = True
            break

    print(
        "5. Decoder gradient: ",
        "PASS" if decoder_gradient else "FAIL"
    )

    assert decoder_gradient

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        "PHASE 8.2 FINAL VERIFICATION: SUCCESS"
    )
    print(
        "=========================================="
    )

    print()
    print("AnomalyVFM+ verified:")
    print("DINOv2")
    print("→ LoRA")
    print("→ Multi-scale feature extraction")
    print("→ Multi-scale fusion")
    print("→ Decoder")
    print("→ Gradient flow")


if __name__ == "__main__":
    main()