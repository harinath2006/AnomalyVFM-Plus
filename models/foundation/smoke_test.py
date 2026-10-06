import torch
import torch.nn as nn


def main():
    print("=== AnomalyVFM+ PyTorch Smoke Test ===")

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Simple test model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )

    # Dummy input
    x = torch.randn(1, 10)

    # Forward pass
    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("PyTorch test: SUCCESS")


if __name__ == "__main__":
    main()