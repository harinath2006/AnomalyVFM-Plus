import torch
from PIL import Image

from models.foundation.dinov2 import DINOv2


def main():

    print("=== DINOv2 Multi-Scale Feature Inspection ===")

    # --------------------------------------------
    # Create test image
    # --------------------------------------------

    image = Image.new(
        "RGB",
        (224, 224),
        "white"
    )

    # --------------------------------------------
    # Load DINOv2
    # --------------------------------------------

    model = DINOv2()
    model.load()

    # --------------------------------------------
    # Prepare input using existing processor
    # --------------------------------------------

    inputs = model.processor(
        images=image,
        return_tensors="pt"
    )

    # --------------------------------------------
    # Get intermediate hidden states
    # --------------------------------------------

    with torch.no_grad():

        outputs = model.model(
            **inputs,
            output_hidden_states=True
        )

    hidden_states = outputs.hidden_states

    print(
        "Number of hidden-state outputs:",
        len(hidden_states)
    )

    # --------------------------------------------
    # Inspect selected layers
    # --------------------------------------------

    selected_layers = [
        0,
        3,
        6,
        9,
        12
    ]

    for layer_index in selected_layers:

        features = hidden_states[layer_index]

        print(
            f"Layer {layer_index}:",
            features.shape
        )

    print()
    print(
        "Expected token shape: [B, 257, 768]"
    )

    print()
    print(
        "Multi-scale feature inspection: SUCCESS"
    )


if __name__ == "__main__":
    main()