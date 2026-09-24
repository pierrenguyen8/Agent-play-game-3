from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PixelBCLSTMConfig:
    """Planet's thin configuration around the PixelBC-style BC architecture.

    The temporal layout follows the PixelBC LSTM policy described upstream:
    visual embedding -> linear projection -> LSTM -> linear action head.
    Recurrence itself is delegated to torch.nn.LSTM rather than reimplemented.
    """

    num_actions: int
    embedding_dim: int = 256
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    encoder_weights: str = "none"
    freeze_encoder: bool = False

    def __post_init__(self) -> None:
        if self.num_actions <= 0:
            raise ValueError("num_actions must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be > 0")
        if self.lstm_num_layers <= 0:
            raise ValueError("lstm_num_layers must be > 0")
        if self.encoder_weights not in ("none", "imagenet"):
            raise ValueError("encoder_weights must be 'none' or 'imagenet'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PixelBCLSTMPolicy:
    """Factory namespace kept import-light for non-ML runtime paths."""

    @staticmethod
    def build(config: PixelBCLSTMConfig):
        try:
            import torch
            from torch import nn
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError(
                "PixelBCLSTMPolicy requires the ml training extra "
                "(torch + torchvision)."
            ) from exc

        weights = (
            ResNet18_Weights.DEFAULT
            if config.encoder_weights == "imagenet"
            else None
        )
        backbone = resnet18(weights=weights)
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()

        if config.freeze_encoder:
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            backbone.eval()

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.encoder = backbone
                self.embedding = nn.Linear(feature_dim, config.embedding_dim)
                self.input_projection = nn.Linear(
                    config.embedding_dim,
                    config.lstm_hidden_size,
                )
                self.lstm = nn.LSTM(
                    input_size=config.lstm_hidden_size,
                    hidden_size=config.lstm_hidden_size,
                    num_layers=config.lstm_num_layers,
                    batch_first=True,
                )
                self.action_head = nn.Linear(
                    config.lstm_hidden_size,
                    config.num_actions,
                )

                # TorchVision ImageNet convention. Keeping this inside the model
                # makes training and future live inference use the same transform.
                self.register_buffer(
                    "image_mean",
                    torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "image_std",
                    torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
                    persistent=False,
                )

            def train(self, mode: bool = True):
                super().train(mode)
                if self.config.freeze_encoder:
                    self.encoder.eval()
                return self

            def forward(self, frames):
                if frames.ndim != 5:
                    raise ValueError(
                        "frames must be [batch, time, channels, height, width]"
                    )
                batch, time, channels, height, width = frames.shape
                if channels != 3:
                    raise ValueError("frames must have exactly 3 RGB channels")

                flat = frames.reshape(batch * time, channels, height, width)
                flat = (flat - self.image_mean) / self.image_std

                if self.config.freeze_encoder:
                    with torch.no_grad():
                        features = self.encoder(flat)
                else:
                    features = self.encoder(flat)

                embedding = self.embedding(features)
                embedding = embedding.reshape(batch, time, -1)
                projected = self.input_projection(embedding)
                temporal, _ = self.lstm(projected)
                return self.action_head(temporal)

            def probabilities(self, frames):
                return torch.sigmoid(self.forward(frames))

        return _Model()
