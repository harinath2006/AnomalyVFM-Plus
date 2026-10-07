import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

from .base import FoundationModel


MODEL_NAME = "facebook/dinov2-base"


class DINOv2(FoundationModel):

    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.processor = None
        self.model = None

    def load(self):
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.eval()

        return self

    def extract_features(self, images, trainable=False):
        """
        Extract spatial visual features.

        Args:
            images:
                PIL image or list of PIL images.

            trainable:
                False -> inference mode
                True  -> gradients enabled for LoRA

        Returns:
            Feature map:
                [B, C, Hf, Wf]
        """

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() first."
            )

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        )

        if trainable:
            outputs = self.model(**inputs)

        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        # Remove CLS token
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        batch_size, num_patches, feature_dim = patch_tokens.shape

        # DINOv2 uses a square patch grid
        grid_size = int(num_patches ** 0.5)

        feature_map = patch_tokens.reshape(
            batch_size,
            grid_size,
            grid_size,
            feature_dim
        )

        # [B, H, W, C] -> [B, C, H, W]
        feature_map = feature_map.permute(
            0,
            3,
            1,
            2
        )

        return feature_map


if __name__ == "__main__":

    print("=== AnomalyVFM+ DINOv2 Module Test ===")

    image = Image.new(
        "RGB",
        (224, 224),
        "white"
    )

    model = DINOv2()

    print("Loading DINOv2...")

    model.load()

    print("Model loaded.")

    features = model.extract_features(image)

    print("Feature shape:", features.shape)

    print("DINOv2 module test: SUCCESS")