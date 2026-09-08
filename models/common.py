import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    """Pad to 'same' shape."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Convolution block with Conv2d + BatchNorm2d + Activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""

    def __init__(self, c1, c2, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = c2
        self.double_conv = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d then DoubleConv."""

    def __init__(self, c1, c2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(c1, c2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with skip-connection concatenation."""

    def __init__(self, c1, c2, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(c1, c2, c1 // 2)
        else:
            self.up = nn.ConvTranspose2d(c1, c1 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(c1, c2)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: input from lower stage (to be upscaled), OR list/tuple [x1, x2]
            x2: skip connection input from corresponding encoder stage
        """
        if x2 is None and isinstance(x1, (list, tuple)):
            x1, x2 = x1[0], x1[1]

        x1 = self.up(x1)

        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x > 0 or diff_y > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channels
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 Convolution to output segmentation logits."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return torch.cat(x, self.d)
        return x
