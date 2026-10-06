from pathlib import Path
import yaml


def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration.

    Returns:
        Dictionary containing the configuration.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":

    config = load_config("configs/baseline.yaml")

    print("=== Configuration Test ===")
    print("Project:", config["project"]["name"])
    print("Experiment:", config["project"]["experiment_id"])
    print("Model:", config["model"]["name"])
    print("Pretrained:", config["model"]["pretrained"])
    print("Image size:", config["input"]["image_size"])
    print("Configuration loaded successfully.")