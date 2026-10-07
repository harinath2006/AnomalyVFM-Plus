import torch
import torch.nn as nn


class ImagePredictor(nn.Module):
    """
    Image-level anomaly predictor.

    Input:
        CLS / summary feature [B, 768]

    Output:
        anomaly logit [B]
    """

    def __init__(self, input_dim=768):
        super().__init__()

        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.classifier(features).squeeze(-1)