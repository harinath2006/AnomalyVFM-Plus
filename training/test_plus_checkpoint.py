import torch
from pathlib import Path

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== AnomalyVFM+ Checkpoint Test ===")

    checkpoint_path = Path(
        "experiments/multiscale/anomaly_vfm_plus_checkpoint.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = AnomalyVFMPlus()

    model.eval()

    # --------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------

    checkpoint = {
        "lora_state": {
            name: parameter.detach().cpu()
            for name, parameter
            in model.foundation_model.named_parameters()
            if parameter.requires_grad
        },

        "fusion_state": model.fusion.state_dict(),

        "decoder_state": model.decoder.state_dict(),

        "config": {
            "lora_rank": 4,
            "lora_alpha": 8,
            "fusion_layers": [3, 6, 9, 12],
            "fusion_channels": 768
        }
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        "Checkpoint saved:",
        checkpoint_path
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    loaded = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print(
        "Checkpoint loaded."
    )

    # --------------------------------------------------
    # Verify sections
    # --------------------------------------------------

    assert "lora_state" in loaded
    assert "fusion_state" in loaded
    assert "decoder_state" in loaded
    assert "config" in loaded

    print(
        "LoRA state: PASS"
    )

    print(
        "Fusion state: PASS"
    )

    print(
        "Decoder state: PASS"
    )

    print(
        "Configuration: PASS"
    )

    # --------------------------------------------------
    # Verify fusion parameters
    # --------------------------------------------------

    print(
        "Fusion parameter tensors:",
        len(loaded["fusion_state"])
    )

    assert len(
        loaded["fusion_state"]
    ) > 0

    # --------------------------------------------------
    # Verify decoder parameters
    # --------------------------------------------------

    print(
        "Decoder parameter tensors:",
        len(loaded["decoder_state"])
    )

    assert len(
        loaded["decoder_state"]
    ) > 0

    # --------------------------------------------------
    # Final verification
    # --------------------------------------------------

    print()
    print(
        "=== AnomalyVFM+ Checkpoint Test: SUCCESS ==="
    )


if __name__ == "__main__":
    main()