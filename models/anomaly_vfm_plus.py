import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from models.foundation.dinov2 import DINOv2
from models.adapters.inject_lora import inject_lora
from models.fusion.multiscale import MultiScaleFeatureFusion
from models.decoder.anomaly_decoder import AnomalyDecoder
from models.decoder.image_predictor import ImagePredictor
from models.decoder.calibration import compute_adaptive_score
from models.decoder.confidence_refinement import refine_anomaly_map


class AnomalyVFMPlus(nn.Module):
    """
    Upgraded AnomalyVFM+ with:
      - Automatic foreground tissue ROI masking (strips MRI/endoscopy black margins)
      - High-resolution progressive decoder
      - Hybrid CLS + spatial-peak image anomaly scoring
    """

    def __init__(
        self,
        lora_rank=8,
        lora_alpha=16,
        decoder_channels=768,
        output_size=224,
        apply_roi_mask=True,
    ):
        super().__init__()
        self.output_size = output_size
        self.apply_roi_mask = apply_roi_mask

        # 1. Foundation model
        foundation = DINOv2()
        foundation.load()
        self.foundation = foundation
        self.foundation_model = foundation.model

        # 2. LoRA (rank 8 captures finer medical texture shifts)
        self.lora_layers = inject_lora(
            self.foundation_model,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            rank=lora_rank,
            alpha=lora_alpha,
        )

        # 3. Multi-scale feature fusion
        self.fusion = MultiScaleFeatureFusion(
            in_channels=768,
            num_scales=4,
            out_channels=768,
        )

        # 4. Progressive high-res pixel decoder
        self.decoder = AnomalyDecoder(
            in_channels=decoder_channels,
            output_size=output_size,
        )

        # 5. Image-level anomaly predictor
        self.predictor = ImagePredictor(input_dim=768)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_foreground_roi(self, images, target_size=None):
        """
        Computes a binary foreground tissue mask [B, 1, H, W] to suppress
        empty dark borders in Brain MRI and circular vignettes in CVC-ClinicDB.
        """
        size = target_size or self.output_size
        masks = []

        if isinstance(images, torch.Tensor):
            imgs = images.detach().cpu()
            if imgs.ndim == 3:
                imgs = imgs.unsqueeze(0)
            gray = imgs.mean(dim=1, keepdim=True)
            gray = F.interpolate(gray, size=(size, size), mode="bilinear", align_corners=False)
            for b in range(gray.shape[0]):
                g = gray[b, 0]
                thresh = g.min() + 0.06 * (g.max() - g.min() + 1e-6)
                m = (g > thresh).float()
                # If almost the entire image is bright (e.g., industrial texture), keep all pixels
                if m.mean() < 0.05:
                    m = torch.ones_like(m)
                masks.append(m.unsqueeze(0))
            return torch.stack(masks).to(self.device)

        if not isinstance(images, (list, tuple)):
            images = [images]

        for img in images:
            if isinstance(img, Image.Image):
                arr = np.asarray(img.convert("L").resize((size, size)), dtype=np.float32) / 255.0
            else:
                arr = np.asarray(img, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=-1)
                if arr.max() > 1.0:
                    arr /= 255.0
            thresh = arr.min() + 0.06 * (arr.max() - arr.min() + 1e-6)
            m = (arr > thresh).astype(np.float32)
            if m.mean() < 0.05:
                m = np.ones_like(m)
            masks.append(torch.from_numpy(m).unsqueeze(0))

        return torch.stack(masks).to(self.device)

    def extract_multiscale_features(self, images, return_summary=False):
        inputs = self.foundation.processor(
            images=images,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.foundation_model(
            **inputs,
            output_hidden_states=True,
        )

        # Combine CLS token with mean patch representation for stronger global sensitivity
        cls_token = outputs.last_hidden_state[:, 0, :]
        mean_patch = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        summary = 0.7 * cls_token + 0.3 * mean_patch

        selected_layers = (3, 6, 9, 12)
        features = []

        for layer_index in selected_layers:
            tokens = outputs.hidden_states[layer_index]
            patch_tokens = tokens[:, 1:, :]
            batch_size, num_patches, channels = patch_tokens.shape
            grid_size = int(num_patches ** 0.5)

            if grid_size * grid_size != num_patches:
                raise ValueError(f"Invalid patch count: {num_patches}")

            feature_map = (
                patch_tokens.reshape(batch_size, grid_size, grid_size, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            features.append(feature_map)

        if return_summary:
            return features, summary
        return features

    def _mask_background_logits(self, anomaly_map, roi_mask):
        if not self.apply_roi_mask or roi_mask is None:
            return anomaly_map
        # Push background logits to -10.0 (sigmoid ~ 4.5e-5) so non-tissue pixels never trigger false positives
        return anomaly_map * roi_mask + (-10.0) * (1.0 - roi_mask)

    def forward(self, images):
        multiscale_features = self.extract_multiscale_features(images)
        fused_features = self.fusion(multiscale_features)
        anomaly_map = self.decoder(fused_features, target_size=self.output_size)

        if self.apply_roi_mask:
            roi_mask = self.compute_foreground_roi(images, target_size=anomaly_map.shape[-1])
            anomaly_map = self._mask_background_logits(anomaly_map, roi_mask)

        return anomaly_map

    def forward_with_score(self, images):
        multiscale_features, summary = self.extract_multiscale_features(
            images, return_summary=True
        )
        fused_features = self.fusion(multiscale_features)
        anomaly_map = self.decoder(fused_features, target_size=self.output_size)

        roi_mask = None
        if self.apply_roi_mask:
            roi_mask = self.compute_foreground_roi(images, target_size=anomaly_map.shape[-1])
            anomaly_map = self._mask_background_logits(anomaly_map, roi_mask)

        cls_logit = self.predictor(summary)

        # Fuse spatial peak logit with CLS logit so localized lesions drive image-level detection
        spatial_peak = anomaly_map.flatten(1).topk(k=16, dim=1).values.mean(dim=1)
        if cls_logit.ndim > spatial_peak.ndim:
            spatial_peak = spatial_peak.unsqueeze(-1)
        image_logit = 0.6 * cls_logit + 0.4 * spatial_peak

        return anomaly_map, image_logit

    def compute_anomaly_score(self, anomaly_map, roi_mask=None):
        return compute_adaptive_score(anomaly_map, roi_mask=roi_mask)

    def predict_with_confidence(self, images, confidence_map):
        anomaly_map, image_logit = self.forward_with_score(images)
        roi_mask = self.compute_foreground_roi(images, target_size=anomaly_map.shape[-1])
        refined_map = refine_anomaly_map(anomaly_map, confidence_map)
        refined_map = self._mask_background_logits(refined_map, roi_mask)

        image_score = torch.sigmoid(image_logit)
        segmentation_score = self.compute_anomaly_score(refined_map, roi_mask=roi_mask)

        return {
            "anomaly_map": anomaly_map,
            "confidence_map": confidence_map,
            "refined_map": refined_map,
            "image_logit": image_logit,
            "image_score": image_score,
            "segmentation_score": segmentation_score,
        }