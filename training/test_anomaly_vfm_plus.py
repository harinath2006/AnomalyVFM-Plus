import torch

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== AnomalyVFM+ Wrapper Test ===")

    model = AnomalyVFMPlus()

    print(
        "LoRA layers:",
        model.lora_layers
    )

    # Synthetic test images.
    images = torch.rand(
        2, 3, 224, 224
    )

    # --------------------------------------------
    # Forward pass
    # --------------------------------------------

    predictions = model(images)

    print(
        "Input shape:",
        images.shape
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    # --------------------------------------------
    # Gradient test
    # --------------------------------------------

    loss = predictions.mean()

    loss.backward()

    lora_gradient_found = False

    for name, parameter in (
        model.foundation_model.named_parameters()
    ):

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

    fusion_gradient_found = False

    for name, parameter in (
        model.fusion.named_parameters()
    ):

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
        "LoRA gradient received:",
        lora_gradient_found
    )

    print(
        "Fusion gradient received:",
        fusion_gradient_found
    )

    assert predictions.shape == (
        2, 1, 16, 16
    )

    assert lora_gradient_found
    assert fusion_gradient_found

    print()
    print(
        "=== AnomalyVFM+ Wrapper Test: SUCCESS ==="
    )


if __name__ == "__main__":
    main()