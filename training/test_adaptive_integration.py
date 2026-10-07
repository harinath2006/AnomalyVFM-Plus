import torch

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== Adaptive Scoring Integration Test ===")

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = AnomalyVFMPlus()

    print(
        "Model created."
    )

    # --------------------------------------------------
    # Input
    # --------------------------------------------------

    images = torch.rand(
        2,
        3,
        224,
        224
    )

    print(
        "Input shape:",
        images.shape
    )

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    anomaly_map, scores = model(
        images
    )

    print(
        "Anomaly map shape:",
        anomaly_map.shape
    )

    print(
        "Adaptive scores:",
        scores
    )

    print(
        "Score shape:",
        scores.shape
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    assert anomaly_map.shape == (
        2,
        1,
        16,
        16
    )

    assert scores.shape == (
        2,
    )

    assert torch.all(
        scores >= 0
    )

    assert torch.all(
        scores <= 1
    )

    print(
        "Score range: PASS"
    )

    # --------------------------------------------------
    # Gradient test
    # --------------------------------------------------

    loss = scores.mean()

    loss.backward()

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
        "LoRA gradient:",
        "PASS" if lora_gradient else "FAIL"
    )

    assert lora_gradient

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print()
    print(
        "=========================================="
    )

    print(
        "ADAPTIVE SCORING INTEGRATION: SUCCESS"
    )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()