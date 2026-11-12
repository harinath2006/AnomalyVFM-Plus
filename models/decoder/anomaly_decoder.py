import torch
import torch.nn as nn
import torch.nn.functional as F
from .scoring import anomaly_score


class UpRefineBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(16, out_ch), num_channels=out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class AnomalyDecoder(nn.Module):
    """
    Progressive multi-stage upsampling decoder.
    Converts [B, C, 16, 16] DINOv2 fused features into [B, 1, 224, 224] logits.
    """

    def __init__(
        self,
        in_channels=768,
        hidden_channels=(256, 128, 64),
        output_size=224,
    ):
        super().__init__()
        self.output_size = output_size

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_channels[0]),
            nn.GELU(),
        )

        # 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = UpRefineBlock(hidden_channels[0], hidden_channels[1])
        self.up2 = UpRefineBlock(hidden_channels[1], hidden_channels[2])
        self.up3 = UpRefineBlock(hidden_channels[2], 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, features, target_size=None):
        x = self.proj(features)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        logits = self.head(x)

        out_size = target_size or self.output_size
        if out_size is not None and logits.shape[-2:] != (out_size, out_size):
            logits = F.interpolate(
                logits,
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False,
            )
        return logits

    def predict(self, features, output_size=224):
        with torch.no_grad():
            anomaly_map = self.forward(features, target_size=output_size)
            score = anomaly_score(anomaly_map)
        return anomaly_map, score