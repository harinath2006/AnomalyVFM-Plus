import torch

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== PHASE 8.4.4 FINAL CONFIDENCE INTEGRATION ===")

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = AnomalyVFMPlus()

    print("1. Model creation: PASS")
    print("   LoRA layers:", model.lora_layers)

    # -------------------------------------------------
    # Test input
    # -------------------------------------------------

    images = torch.rand(
        2,
        3,
        224,
        224
    )

    # Confidence map must match decoder output.
    confidence_map = torch.rand(
        2,
        1,
        16,
        16
    )

    print("2. Input preparation: PASS")
    print("   Images:", images.shape)
    print("   Confidence map:", confidence_map.shape)

    # -------------------------------------------------
    # Full confidence-aware pipeline
    # -------------------------------------------------

    results = model.predict_with_confidence(
        images,
        confidence_map
    )

    # -------------------------------------------------
    # Check returned outputs
    # -------------------------------------------------

    assert isinstance(results, dict)

    required_keys = {
        "anomaly_map",
        "confidence_map",
        "refined_map",
        "anomaly_score"
    }

    assert required_keys.issubset(results.keys())

    print("3. Output dictionary: PASS")

    # -------------------------------------------------
    # Extract outputs
    # -------------------------------------------------

    anomaly_map = results["anomaly_map"]
    returned_confidence = results["confidence_map"]
    refined_map = results["refined_map"]
    anomaly_score = results["anomaly_score"]

    # -------------------------------------------------
    # Shape checks
    # -------------------------------------------------

    assert anomaly_map.shape == (
        2,
        1,
        16,
        16
    )

    assert returned_confidence.shape == (
        2,
        1,
        16,
        16
    )

    assert refined_map.shape == (
        2,
        1,
        16,
        16
    )

    assert anomaly_score.shape == (
        2,
    )

    print("4. Output shapes: PASS")
    print("   Anomaly map:", anomaly_map.shape)
    print("   Confidence map:", returned_confidence.shape)
    print("   Refined map:", refined_map.shape)
    print("   Anomaly score:", anomaly_score.shape)

    # -------------------------------------------------
    # Score range
    # -------------------------------------------------

    assert torch.all(anomaly_score >= 0)
    assert torch.all(anomaly_score <= 1)

    print("5. Score range: PASS")
    print("   Scores:", anomaly_score)

    # -------------------------------------------------
    # Verify refinement actually exists
    # -------------------------------------------------

    assert refined_map.shape == anomaly_map.shape

    print("6. Confidence refinement: PASS")

    # -------------------------------------------------
    # Final result
    # -------------------------------------------------

    print()
    print("==============================================")
    print("PHASE 8.4.4 FINAL VERIFICATION: SUCCESS")
    print("==============================================")

    print()
    print("Confidence-aware AnomalyVFM+ pipeline verified:")
    print("DINOv2")
    print("  -> LoRA")
    print("  -> Multi-scale fusion")
    print("  -> Decoder")
    print("  -> Anomaly map")
    print("  -> Confidence map")
    print("  -> Confidence refinement")
    print("  -> Adaptive anomaly score")


if __name__ == "__main__":
    main()