from models.foundation import create_foundation_model
from configs.loader import load_config


def main():
    print("=== DINOv2 Attention Module Inspection ===")

    config = load_config("configs/baseline.yaml")

    model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading DINOv2...")
    model.load()

    print("\nAttention-related modules:\n")

    for name, module in model.model.named_modules():
        if "attention" in name.lower():
            print(name, "->", type(module).__name__)


if __name__ == "__main__":
    main()