import torch.nn as nn

from .lora import LoRALinear


def inject_lora(
    model,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    rank=4,
    alpha=8
):
    """
    Inject LoRA adapters into selected Linear layers.

    Args:
        model:
            DINOv2 model.

        target_modules:
            Attention projections to adapt.

        rank:
            LoRA rank.

        alpha:
            LoRA scaling parameter.

    Returns:
        Number of LoRA layers injected.
    """
        # Freeze the entire pretrained model first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    injected = 0

    for name, module in list(model.named_modules()):

        # Only target the requested attention projections
        if not any(name.endswith(target) for target in target_modules):
            continue

        if not isinstance(module, nn.Linear):
            continue

        # Find parent module
        parent_name, child_name = name.rsplit(".", 1)

        parent = model.get_submodule(parent_name)

        # Replace original Linear with LoRA version
        setattr(
            parent,
            child_name,
            LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha
            )
        )

        injected += 1

    return injected
if __name__ == "__main__":
    from models.foundation import create_foundation_model

    print("=== LoRA Injection Test ===")

    foundation = create_foundation_model("dinov2")

    print("Loading DINOv2...")
    foundation.load()

    print("Injecting LoRA...")

    count = inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    print("LoRA layers injected:", count)

    print("\nChecking injected layers:")

    for name, module in foundation.model.named_modules():
        if isinstance(module, LoRALinear):
            print(name, "-> LoRALinear")

    print("\nLoRA injection test: SUCCESS")