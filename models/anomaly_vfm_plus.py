import torch
import torch.nn as nn

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.fusion.multiscale import MultiScaleFeatureFusion
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.decoder.image_predictor import ImagePredictor
from models.decoder.calibration import compute_adaptive_score
from models.decoder.confidence_refinement import refine_anomaly_map


class AnomalyVFMPlus(nn.Module):
    """
    AnomalyVFM+ model.

    Architecture:

        Image
          |
        DINOv2
          |
        LoRA
          |
        +-----------------------------+
        |                             |
        v                             v
    Patch tokens                  CLS token
        |                             |
        v                             v
    Multi-scale                 Image Predictor
      fusion                         |
        |                             |
        v                             v
     Decoder                    Image anomaly logit
        |
        v
    Anomaly map
    """

    def __init__(
        self,
        lora_rank=4,
        lora_alpha=8,
        decoder_channels=768,
    ):
        super().__init__()

        # --------------------------------------------------
        # 1. Foundation model
        # --------------------------------------------------

        foundation = DINOv2()
        foundation.load()

        self.foundation = foundation
        self.foundation_model = foundation.model

        # --------------------------------------------------
        # 2. LoRA
        # --------------------------------------------------

        self.lora_layers = inject_lora(
            self.foundation_model,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ),
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # --------------------------------------------------
        # 3. Multi-scale feature fusion
        # --------------------------------------------------

        self.fusion = MultiScaleFeatureFusion(
            in_channels=768,
            num_scales=4,
            out_channels=768,
        )

        # --------------------------------------------------
        # 4. Pixel-level anomaly decoder
        # --------------------------------------------------

        self.decoder = AnomalyDecoder(
            in_channels=decoder_channels
        )

        # --------------------------------------------------
        # 5. Image-level anomaly predictor
        # --------------------------------------------------

        self.predictor = ImagePredictor(
            input_dim=768
        )

    # ======================================================
    # FEATURE EXTRACTION
    # ======================================================

    def extract_multiscale_features(
        self,
        images,
        return_summary=False,
    ):
        """
        Extract multi-scale patch features and CLS feature.

        Returns:

            features

        or:

            features, summary
        """

        inputs = self.foundation.processor(
            images=images,
            return_tensors="pt",
        )

        outputs = self.foundation_model(
            **inputs,
            output_hidden_states=True,
        )

        # --------------------------------------------------
        # CLS / summary feature
        # --------------------------------------------------

        summary = outputs.last_hidden_state[:, 0, :]

        # --------------------------------------------------
        # Multi-scale patch features
        # --------------------------------------------------

        selected_layers = (
            3,
            6,
            9,
            12,
        )

        features = []

        for layer_index in selected_layers:

            tokens = outputs.hidden_states[layer_index]

            # Remove CLS token
            patch_tokens = tokens[:, 1:, :]

            batch_size = patch_tokens.shape[0]
            num_patches = patch_tokens.shape[1]
            channels = patch_tokens.shape[2]

            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"Invalid patch count: {num_patches}"
                )

            feature_map = patch_tokens.reshape(
                batch_size,
                grid_size,
                grid_size,
                channels,
            )

            feature_map = feature_map.permute(
                0,
                3,
                1,
                2,
            )

            features.append(feature_map)

        if return_summary:
            return features, summary

        return features

    # ======================================================
    # NORMAL FORWARD
    # ======================================================

    def forward(self, images):
        """
        Returns only the anomaly map.

        Kept this way so existing code does not break.
        """

        multiscale_features = (
            self.extract_multiscale_features(images)
        )

        fused_features = self.fusion(
            multiscale_features
        )

        anomaly_map = self.decoder(
            fused_features
        )

        return anomaly_map

    # ======================================================
    # FORWARD WITH IMAGE SCORE
    # ======================================================

    def forward_with_score(self, images):
        """
        Full forward pass.

        Returns:

            anomaly_map
            image_logit
        """

        multiscale_features, summary = (
            self.extract_multiscale_features(
                images,
                return_summary=True,
            )
        )

        # Pixel-level branch
        fused_features = self.fusion(
            multiscale_features
        )

        anomaly_map = self.decoder(
            fused_features
        )

        # Image-level branch
        image_logit = self.predictor(
            summary
        )

        return anomaly_map, image_logit

    # ======================================================
    # ANOMALY SCORE
    # ======================================================

    def compute_anomaly_score(
        self,
        anomaly_map,
    ):
        """
        Legacy/adaptive segmentation score.

        This is kept for analysis and visualization.

        The main classification score should now come
        from the ImagePredictor.
        """

        return compute_adaptive_score(
            anomaly_map
        )

    # ======================================================
    # CONFIDENCE REFINEMENT
    # ======================================================

    def predict_with_confidence(
        self,
        images,
        confidence_map,
    ):
        """
        Full prediction with confidence refinement.
        """

        anomaly_map, image_logit = (
            self.forward_with_score(images)
        )

        refined_map = refine_anomaly_map(
            anomaly_map,
            confidence_map,
        )

        image_score = torch.sigmoid(
            image_logit
        )

        segmentation_score = (
            self.compute_anomaly_score(
                refined_map
            )
        )

        return {
            "anomaly_map": anomaly_map,
            "confidence_map": confidence_map,
            "refined_map": refined_map,
            "image_logit": image_logit,
            "image_score": image_score,
            "segmentation_score": segmentation_score,
        }