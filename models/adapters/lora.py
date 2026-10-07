import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Linear layer with a low-rank LoRA adaptation.

    Original layer:
        y = W(x)

    With LoRA:
        y = W(x) + scale * B(A(x))
    """

    def __init__(self, original_layer, rank=4, alpha=8):
        super().__init__()

        if not isinstance(original_layer, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear layer.")

        self.original = original_layer

        # Freeze the pretrained weights
        for param in self.original.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Linear(
            original_layer.in_features,
            rank,
            bias=False
        )

        self.lora_B = nn.Linear(
            rank,
            original_layer.out_features,
            bias=False
        )

        # Standard LoRA initialization:
        # A is random, B starts at zero.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5)
        )

        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original(x)

        lora_output = self.lora_B(
            self.lora_A(x)
        )

        return original_output + self.scale * lora_output