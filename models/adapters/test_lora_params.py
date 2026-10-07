from models.foundation import create_foundation_model
from models.adapters.inject_lora import inject_lora
from models.adapters.lora import LoRALinear


def main():
    print("=== LoRA Parameter Test ===")

    foundation = create_foundation_model("dinov2")

    print("Loading DINOv2...")
    foundation.load()

    print("Injecting LoRA...")
    count = inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    print("LoRA layers:", count)

    trainable = 0
    frozen = 0

    for name, parameter in foundation.model.named_parameters():

        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()

    print("\nTrainable parameters:", trainable)
    print("Frozen parameters:", frozen)

    print("\nTrainable parameter names:")

    for name, parameter in foundation.model.named_parameters():
        if parameter.requires_grad:
            print(name)

    print("\nLoRA parameter test: SUCCESS")


if __name__ == "__main__":
    main()