import os
import sys
import logging
from copy import deepcopy
from pathlib import Path
import yaml
import torch
import torch.nn as nn

from models.common import (
    Conv,
    DoubleConv,
    Down,
    Up,
    OutConv,
    Concat
)

logger = logging.getLogger(__name__)


def parse_model(d, ch):
    """
    Parse declarative model architecture YAML dictionary into PyTorch ModuleList/Sequential.

    Args:
        d (dict): Model configuration dictionary containing 'backbone' and 'head'
        ch (list of int): Input channel history, starting with [in_channels]

    Returns:
        model (nn.Sequential): Executable sequential container of model layers
        save (list of int): Layer output indices to cache for skip-connections
    """
    logger.info(f"{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<30}{'arguments'}")
    nc = d.get('nc', 1)
    gd = d.get('depth_multiple', 1.0)
    gw = d.get('width_multiple', 1.0)

    layers, save = [], []
    c2 = ch[-1]  # output channel tracker

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        # Resolve module class
        if isinstance(m, str):
            m = eval(m)  # resolve string name to class in scope

        # Evaluate string arguments if any
        for j, a in enumerate(args):
            if isinstance(a, str):
                try:
                    args[j] = eval(a)
                except Exception:
                    pass

        n = max(round(n * gd), 1) if n > 1 else n

        # Layer specific channel wiring
        if m in [DoubleConv, Down, Conv]:
            c1 = ch[f]
            c2 = args[0]
            if not (i == 0 and m is Conv and c2 <= 4):  # preserve input projection if present
                c2 = max(round(c2 * gw / 8) * 8, 8) if gw != 1.0 else c2
            args = [c1, c2, *args[1:]]

        elif m is Up:
            # f is [lower_idx, skip_idx]
            c1_lower = ch[f[0]]
            c2 = args[0]
            c2 = max(round(c2 * gw / 8) * 8, 8) if gw != 1.0 else c2
            args = [c1_lower, c2, *args[1:]]

        elif m is OutConv:
            c1 = ch[f]
            c2 = nc  # Dynamically driven by top-level nc configuration
            args = [c1, c2, *args[1:]]

        elif m is Concat:
            c2 = sum([ch[x] for x in f])

        else:
            c1 = ch[f]
            c2 = args[0] if len(args) > 0 else c1
            if len(args) > 0 and c2 != nc:
                c2 = max(round(c2 * gw / 8) * 8, 8) if gw != 1.0 else c2
            args = [c1, c2, *args[1:]]

        # Construct layer instance
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info(f"{i:>3}{str(f):>18}{n:>3}{np:>10.0f}  {t:<30}{str(args)}")

        # Track save indices for skip-connections
        if isinstance(f, int):
            if f != -1:
                save.append(f % i)
        elif isinstance(f, (list, tuple)):
            save.extend([x % i for x in f if x != -1])

        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    return nn.Sequential(*layers), sorted(list(set(save)))


class Model(nn.Module):
    """
    Multi-Modal Declarative Segmentation Model with dynamic input projection.
    """

    def __init__(self, cfg='models/architectures/unet.yaml', ch=3, nc=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = deepcopy(cfg)
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, 'r') as f:
                self.yaml = yaml.safe_load(f)

        # Override class count if requested
        if nc is not None:
            self.yaml['nc'] = nc

        self.in_channels = ch
        self.nc = self.yaml.get('nc', 1)

        # Construct architecture via model_parser
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[self.in_channels])

        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        """
        Forward pass through the parsed sequential layers.

        Args:
            x (torch.Tensor): multi-channel input tensor (B, C_in, H, W)

        Returns:
            torch.Tensor: segmentation logits (B, nc, H, W)
        """
        y = []
        for m in self.model:
            if m.f != -1:
                if isinstance(m.f, int):
                    x_in = y[m.f]
                else:
                    x_in = [x if j == -1 else y[j] for j in m.f]
            else:
                x_in = x

            x = m(x_in)
            y.append(x if m.i in self.save else None)

        return x

    def initialize_weights(self):
        """Kaiming normal initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def info(self, verbose=False, img_size=(512, 512)):
        """Print model parameters and FLOPs."""
        n_p = sum(x.numel() for x in self.parameters())
        n_g = sum(x.numel() for x in self.parameters() if x.requires_grad)
        logger.info(f"Model Summary: {len(list(self.modules()))} layers, {n_p:,} parameters, {n_g:,} gradients")
        print(f"Model Summary: {len(list(self.modules()))} layers, {n_p:,} parameters, {n_g:,} gradients")
        return n_p, n_g
