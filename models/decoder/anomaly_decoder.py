import torch
import torch.nn as nn
from .scoring import anomaly_score

class AnomalyDecoder(nn.Module):
    """
    Converts foundation-model features into an anomaly map.

    Input:
        [B, C, H, W]

    Output:
        [B, 1, H, W]
    """

    def __init__(self,in_channels=768,hidden_channels=(256, 128, 64)):
        super().__init__()

        self.decoder = nn.Sequential(nn.Conv2d(in_channels,hidden_channels[0],kernel_size=3,padding=1),
        nn.ReLU(),

        nn.Conv2d(
        hidden_channels[0],
        hidden_channels[1],
        kernel_size=3,
        padding=1
        ),
        nn.ReLU(),

        nn.Conv2d(
        hidden_channels[1],
        hidden_channels[2],
        kernel_size=3,
        padding=1
        ),
        nn.ReLU(),

        nn.Conv2d(
        hidden_channels[2],
        1,
        kernel_size=1
        )
    )

    def forward(self, features):
        return self.decoder(features)
    
    def predict(self, features, output_size=224):
        """
        Generate anomaly map and image-level anomaly score.

        This method is intended for inference.
        Gradients are disabled to reduce memory usage.
        """

        with torch.no_grad():
            anomaly_map = self.forward(features)

            anomaly_map = nn.functional.interpolate(
                anomaly_map,
                size=(output_size, output_size),
                mode="bilinear",
                align_corners=False
            )

            score = anomaly_score(anomaly_map)

        return anomaly_map, score

if __name__ == "__main__":
    print("=== DINOv2 → Anomaly Decoder Test ===")

    from models.foundation import create_foundation_model
    from configs.loader import load_config
    from PIL import Image

    config = load_config("configs/baseline.yaml")

    # Create foundation model from configuration
    foundation_model = create_foundation_model(
        config["model"]["name"]
    )

    print("Loading foundation model...")
    foundation_model.load()

    # Create a test image
    image = Image.new("RGB", (224, 224), "white")

    print("Extracting foundation features...")
    features = foundation_model.extract_features(image)

    print("Foundation feature shape:", features.shape)

    # Create decoder
    decoder_config = config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(decoder_config["hidden_channels"])
    )

    print("Generating anomaly map...")
    anomaly_map, score = decoder.predict(features, output_size=decoder_config["output_size"])

    print("Anomaly map shape:", anomaly_map.shape)
    print("Anomaly score:", score)
    print("End-to-end decoder test: SUCCESS")