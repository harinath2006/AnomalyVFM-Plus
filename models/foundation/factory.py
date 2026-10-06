from .dinov2 import DINOv2


def create_foundation_model(model_name):
    """
    Create a foundation model based on its name.
    """

    if model_name.lower() == "dinov2":
        return DINOv2()

    raise ValueError(
        f"Unsupported foundation model: {model_name}"
    )