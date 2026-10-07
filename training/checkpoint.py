from pathlib import Path
import torch


def save_checkpoint(
    path,
    foundation_model,
    decoder,
    optimizer,
    step,
    config,
    fusion=None
):
    """
    Save experiment state.

    Saves:
    - LoRA parameters
    - Decoder parameters
    - Multi-scale fusion parameters (if provided)
    - Optimizer state
    - Training step
    - Configuration
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # LoRA state
    # --------------------------------------------------

    lora_state = {
        name: parameter.detach().cpu()
        for name, parameter in foundation_model.named_parameters()
        if parameter.requires_grad
    }

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    checkpoint = {
        "step": step,
        "lora_state": lora_state,
        "decoder_state": decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
    }

    # --------------------------------------------------
    # Multi-scale fusion
    # --------------------------------------------------

    if fusion is not None:
        checkpoint["fusion_state"] = {
            name: parameter.detach().cpu()
            for name, parameter in fusion.state_dict().items()
        }

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    torch.save(
        checkpoint,
        path
    )

    print("Checkpoint saved:", path)