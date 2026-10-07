import torch
import torch.nn as nn


class MultiScaleFeatureExtractor(nn.Module):
    """
    Converts selected DINOv2 hidden states into
    spatial feature maps for multi-scale fusion.

    Input:
        hidden_states:
            list of tensors
            each [B, 257, 768]

    Output:
        list of feature maps
            each [B, 768, 16, 16]
    """

    def __init__(
        self,
        selected_layers=(3, 6, 9, 12)
    ):
        super().__init__()

        self.selected_layers = selected_layers

    def forward(self, hidden_states):

        features = []

        for layer_index in self.selected_layers:

            tokens = hidden_states[layer_index]

            # Remove CLS token.
            patch_tokens = tokens[:, 1:, :]

            batch_size = patch_tokens.shape[0]
            num_patches = patch_tokens.shape[1]
            channels = patch_tokens.shape[2]

            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"Cannot reshape {num_patches} patches "
                    "into a square feature map."
                )

            feature_map = patch_tokens.reshape(
                batch_size,
                grid_size,
                grid_size,
                channels
            )

            feature_map = feature_map.permute(
                0,
                3,
                1,
                2
            )

            features.append(feature_map)

        return features
    
class MultiScaleFeatureFusion(nn.Module):
    """
    Fuse multi-scale feature maps into a single representation.

    Input:
        list of feature maps:
        [B, 768, 16, 16] x 4

    Output:
        [B, 768, 16, 16]
    """

    def __init__(
        self,
        in_channels=768,
        num_scales=4,
        out_channels=768
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels * num_scales,
                out_channels,
                kernel_size=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):

        if len(features) != 4:
            raise ValueError(
                f"Expected 4 feature scales, got {len(features)}"
            )

        # Make sure all feature maps have identical shapes.
        reference_shape = features[0].shape

        for feature in features:

            if feature.shape != reference_shape:
                raise ValueError(
                    "All feature maps must have the same shape."
                )

        # Concatenate along channel dimension.
        combined = torch.cat(
            features,
            dim=1
        )

        # Learnable fusion.
        fused = self.fusion(combined)

        return fused

if __name__ == "__main__":

    print("=== Multi-Scale Feature Fusion Test ===")

    # Simulated multi-scale features.
    features = [
        torch.randn(2, 768, 16, 16)
        for _ in range(4)
    ]

    fusion = MultiScaleFeatureFusion(
        in_channels=768,
        num_scales=4,
        out_channels=768
    )

    fused = fusion(features)

    print(
        "Number of input scales:",
        len(features)
    )

    print(
        "Input feature shape:",
        features[0].shape
    )

    print(
        "Fused feature shape:",
        fused.shape
    )

    print(
        "Fused feature requires grad:",
        fused.requires_grad
    )

    print(
        "Multi-scale fusion test: SUCCESS"
    )