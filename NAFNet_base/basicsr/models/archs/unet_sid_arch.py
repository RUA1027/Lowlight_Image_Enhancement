"""U-Net architecture tailored for the SID low-light restoration baseline."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetSID(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: Sequence[int] | None = None,
        bilinear: bool = True,
    ):
        super().__init__()

        if depth is None:
            depth = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        if len(depth) < 2:
            raise ValueError("Depth sequence for UNetSID must contain at least two stages.")

        channels = list(depth)
        self.inc = DoubleConv(in_channels, channels[0])
        self.down_layers = nn.ModuleList()
        for idx in range(len(channels) - 1):
            self.down_layers.append(Down(channels[idx], channels[idx + 1]))

        factor = 2 if bilinear else 1
        self.bottom = DoubleConv(channels[-1], channels[-1] * factor)

        self.up_layers = nn.ModuleList()
        reversed_channels = [channels[-1] * factor] + list(reversed(channels[:-1]))
        out_channels_list = list(reversed(channels))
        for idx in range(len(reversed_channels) - 1):
            self.up_layers.append(
                Up(reversed_channels[idx], out_channels_list[idx + 1] // (2 if bilinear else 1), bilinear=bilinear)
            )

        self.outc = nn.Conv2d(out_channels_list[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inc(x)
        encoder_feats = [x0]
        out = x0
        for layer in self.down_layers:
            out = layer(out)
            encoder_feats.append(out)

        out = self.bottom(out)

        for layer, skip in zip(self.up_layers, reversed(encoder_feats[:-1])):
            out = layer(out, skip)

        return self.outc(out)


def UNetSIDModel(**kwargs) -> UNetSID:
    return UNetSID(**kwargs)
