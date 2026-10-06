from abc import ABC, abstractmethod
import torch


class FoundationModel(ABC):
    """
    Common interface for all Vision Foundation Models.

    Every foundation model used in AnomalyVFM+
    should follow this interface.
    """

    @abstractmethod
    def load(self):
        """Load the pretrained model."""
        pass

    @abstractmethod
    def extract_features(self, images: torch.Tensor):
        """
        Extract spatial visual features.

        Input:
            images: Tensor [B, 3, H, W]

        Output:
            feature_map: Tensor [B, C, Hf, Wf]
        """
        pass