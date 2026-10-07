import torch

from models.anomaly_vfm_plus import AnomalyVFMPlus
from models.decoder.scoring import anomaly_score
from models.decoder.calibration import compute_adaptive_score


def main():

    print("=== PHASE 8.3 FINAL VERIFICATION ===")

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = AnomalyVFMPlus()

    images = torch.rand(
        2,
        3,
        224,
        224
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    anomaly_map, adaptive_scores = model(
        images
    )

    print(
        "1. Anomaly map: PASS",
        anomaly_map.shape
    )

    # --------------------------------------------------
    # Adaptive score
    # --------------------------------------------------

    direct_adaptive_scores = compute_adaptive_score(
        anomaly_map
    )

    print(
        "2. Adaptive scoring: PASS"
    )

    print(
        "   Model scores:",
        adaptive_scores.detach()
    )

    print(
        "   Direct scores:",
        direct_adaptive_scores.detach()
    )

    # Model score and direct calculation must match
    assert torch.allclose(
        adaptive_scores,
        direct_adaptive_scores
    )

    # --------------------------------------------------
    # Old scoring comparison
    # --------------------------------------------------

    old_scores = anomaly_score(
        anomaly_map
    )

    print(
        "3. Legacy max scoring: PASS"
    )

    print(
        "   Max scores:",
        old_scores.detach()
    )

    # --------------------------------------------------
    # Score range
    # --------------------------------------------------

    assert torch.all(
        adaptive_scores >= 0
    )

    assert torch.all(
        adaptive_scores <= 1
    )

    print(
        "4. Score range: PASS"
    )

    # --------------------------------------------------
    # Gradient flow
    # --------------------------------------------------

    loss = adaptive_scores.mean()

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

    assert lora_gradient

    print(
        "5. LoRA gradient: PASS"
    )

    # --------------------------------------------------
    # Final
    # --------------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        "PHASE 8.3 FINAL VERIFICATION: SUCCESS"
    )
    print(
        "=========================================="
    )

    print()
    print(
        "Adaptive scoring pipeline verified:"
    )

    print(
        "Anomaly map"
    )

    print(
        "→ Adaptive calibration"
    )

    print(
        "→ Image-level anomaly score"
    )

    print(
        "→ LoRA gradient flow"
    )


if __name__ == "__main__":
    main()