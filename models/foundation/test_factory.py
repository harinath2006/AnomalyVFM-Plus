from configs.loader import load_config
from models.foundation import create_foundation_model


def main():
    config = load_config("configs/baseline.yaml")

    model_name = config["model"]["name"]

    model = create_foundation_model(model_name)

    print("=== Foundation Model Factory Test ===")
    print("Configured model:", model_name)
    print("Created model:", type(model).__name__)
    print("Factory test: SUCCESS")


if __name__ == "__main__":
    main()