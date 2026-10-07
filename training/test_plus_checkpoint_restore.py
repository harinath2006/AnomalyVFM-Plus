import torch
from pathlib import Path

from models.anomaly_vfm_plus import AnomalyVFMPlus


def main():

    print("=== AnomalyVFM+ Checkpoint Restore Test ===")

    checkpoint_path = Path(
        "experiments/multiscale/anomaly_vfm_plus_checkpoint.pt"
    )

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print("Checkpoint loaded.")

    # --------------------------------------------------
    # Create fresh model
    # --------------------------------------------------

    model = AnomalyVFMPlus()

    print(
        "Fresh AnomalyVFM+ model created."
    )

    # --------------------------------------------------
    # Restore LoRA
    # --------------------------------------------------

    current_lora_state = {
        name: parameter
        for name, parameter
        in model.foundation_model.named_parameters()
        if parameter.requires_grad
    }

    restored_lora = 0

    for name, saved_parameter in checkpoint[
        "lora_state"
    ].items():

        if name not in current_lora_state:
            raise KeyError(
                f"LoRA parameter not found: {name}"
            )

        current_lora_state[name].data.copy_(
            saved_parameter
        )

        restored_lora += 1

    print(
        "LoRA tensors restored:",
        restored_lora
    )

    # --------------------------------------------------
    # Restore fusion
    # --------------------------------------------------

    model.fusion.load_state_dict(
        checkpoint["fusion_state"]
    )

    print(
        "Fusion weights restored."
    )

    # --------------------------------------------------
    # Restore decoder
    # --------------------------------------------------

    model.decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    print(
        "Decoder weights restored."
    )

    # --------------------------------------------------
    # Verify LoRA restoration
    # --------------------------------------------------

    lora_match = True

    for name, saved_parameter in checkpoint[
        "lora_state"
    ].items():

        current_parameter = dict(
            model.foundation_model.named_parameters()
        )[name]

        if not torch.equal(
            current_parameter.detach().cpu(),
            saved_parameter
        ):
            lora_match = False
            break

    print(
        "LoRA restoration:",
        "PASS" if lora_match else "FAIL"
    )

    # --------------------------------------------------
    # Verify fusion restoration
    # --------------------------------------------------

    fusion_match = True

    current_fusion = model.fusion.state_dict()

    for name, saved_parameter in checkpoint[
        "fusion_state"
    ].items():

        if not torch.equal(
            current_fusion[name].cpu(),
            saved_parameter.cpu()
        ):
            fusion_match = False
            break

    print(
        "Fusion restoration:",
        "PASS" if fusion_match else "FAIL"
    )

    # --------------------------------------------------
    # Verify decoder restoration
    # --------------------------------------------------

    decoder_match = True

    current_decoder = model.decoder.state_dict()

    for name, saved_parameter in checkpoint[
        "decoder_state"
    ].items():

        if not torch.equal(
            current_decoder[name].cpu(),
            saved_parameter.cpu()
        ):
            decoder_match = False
            break

    print(
        "Decoder restoration:",
        "PASS" if decoder_match else "FAIL"
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert lora_match
    assert fusion_match
    assert decoder_match

    print()
    print(
        "=============================================="
    )
    print(
        "ANOMALYVFM+ CHECKPOINT RESTORE: SUCCESS"
    )
    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()