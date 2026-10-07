import torch

from models.foundation import create_foundation_model
from models.adapters.inject_lora import inject_lora


def main():

    print("=== LoRA Gradient Test ===")

    foundation = create_foundation_model("dinov2")

    print("Loading DINOv2...")
    foundation.load()

    print("Injecting LoRA...")
    inject_lora(
        foundation.model,
        rank=4,
        alpha=8
    )

    # Simple test image
    image = torch.rand(3, 224, 224)

    # Convert tensor to PIL because our current processor expects images
    from torchvision.transforms.functional import to_pil_image

    image = to_pil_image(image)

    # IMPORTANT:
    # trainable=True enables gradient flow
    features = foundation.extract_features(
        image,
        trainable=True
    )

    print("Feature shape:", features.shape)
    print("Feature requires_grad:", features.requires_grad)

    # Create a simple scalar
    loss = features.mean()

    print("Loss:", loss.item())

    # Backpropagation
    loss.backward()

    # Check whether LoRA received gradients

    gradient_found = False  

    for name, parameter in foundation.model.named_parameters():

        if "lora_B" in name and parameter.grad is not None:

            gradient_value = parameter.grad.abs().mean().item()

            print(
                "Gradient received:",
                name,
                "->",
                gradient_value
            )

            if gradient_value > 0:
                gradient_found = True

            break


    if gradient_found:
        print("\nLoRA gradient test: SUCCESS")
    else:   
        print("\nLoRA gradient test: FAILED")

    


if __name__ == "__main__":
    main()