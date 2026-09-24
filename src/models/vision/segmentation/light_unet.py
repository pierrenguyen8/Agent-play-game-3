from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class LightUNet(nn.Module):
    """Planet platform U-Net compatible with the existing trained checkpoint.

    The original checkpoint stores decoder parameters under ``dec4`` ...
    ``dec1``. Keep these module names stable: checkpoint compatibility is part
    of the production model contract.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.enc3 = ConvBlock(b * 2, b * 4)
        self.enc4 = ConvBlock(b * 4, b * 8)
        self.bottleneck = ConvBlock(b * 8, b * 16)
        self.pool = nn.MaxPool2d(2)

        # IMPORTANT: these names match the existing Planet checkpoint.
        self.dec4 = ConvBlock(b * 16 + b * 8, b * 8)
        self.dec3 = ConvBlock(b * 8 + b * 4, b * 4)
        self.dec2 = ConvBlock(b * 4 + b * 2, b * 2)
        self.dec1 = ConvBlock(b * 2 + b, b)
        self.head = nn.Conv2d(b, out_channels, 1)

    @staticmethod
    def _decode(x: Tensor, skip: Tensor, block: nn.Module) -> Tensor:
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return block(torch.cat((x, skip), dim=1))

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))

        x = self._decode(x, e4, self.dec4)
        x = self._decode(x, e3, self.dec3)
        x = self._decode(x, e2, self.dec2)
        x = self._decode(x, e1, self.dec1)
        return self.head(x)
