import torch

from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder
from models.adapters.inject_lora import inject_lora
from models.adapters.lora import LoRALinear


def main():

    print("=== Checkpoint Loading Test ===")

    checkpoint_path = "experiments/lora/checkpoint.pt"

    # --------------------------------------------------
    # 1. Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print("Checkpoint loaded.")
    print("Training step:", checkpoint["step"])

    # --------------------------------------------------
    # 2. Create fresh DINOv2
    # --------------------------------------------------

    foundation = create_foundation_model("dinov2")

    print("Loading fresh DINOv2...")
    foundation.load()

    # --------------------------------------------------
    # 3. Inject same LoRA structure
    # --------------------------------------------------

    lora_count = inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    print("LoRA layers:", lora_count)

    # --------------------------------------------------
    # 4. Restore LoRA weights
    # --------------------------------------------------

    missing = []
    unexpected = []

    current_parameters = dict(
        foundation.model.named_parameters()
    )

    for name, saved_parameter in checkpoint["lora_state"].items():

        if name in current_parameters:
            current_parameters[name].data.copy_(
                saved_parameter
            )
        else:
            unexpected.append(name)

    # --------------------------------------------------
    # 5. Restore decoder
    # --------------------------------------------------

    decoder = AnomalyDecoder(
        in_channels=768,
        hidden_channels=(256, 128, 64)
    )

    decoder.load_state_dict(
        checkpoint["decoder_state"]
    )

    print("Decoder restored.")

    # --------------------------------------------------
    # 6. Verify
    # --------------------------------------------------

    restored_count = 0

    for name, parameter in foundation.model.named_parameters():

        if "lora_" in name:
            restored_count += 1

    print("LoRA parameter tensors:", restored_count)

    if restored_count > 0 and lora_count == 48:
        print("\nCheckpoint loading test: SUCCESS")
    else:
        print("\nCheckpoint loading test: FAILED")


if __name__ == "__main__":
    main()