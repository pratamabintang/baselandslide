import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Foreground-only Soft Dice Loss calculated per-image and averaged across the batch.

    Specification:
    - Model output: 2 classes (B, 2, H, W) -> class 0: background, class 1: foreground/landslide
    - Softmax applied along dim=1 to extract foreground probabilities: probs[:, 1, :, :]
    - Dice calculated per-image across spatial dimensions (H, W), then averaged over batch B.
    - Smoothing parameter: smooth=1.0
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Raw model output logits of shape (B, 2, H, W)
            targets (torch.Tensor): Ground truth class indices of shape (B, H, W) or (B, 1, H, W)
                                    where 0 = background, 1 = landslide/foreground

        Returns:
            torch.Tensor: Scalar Dice loss averaged across batch
        """
        # 1. Softmax over class dimension and extract foreground probabilities (class 1)
        probs = F.softmax(logits, dim=1)
        foreground_prob = probs[:, 1, :, :]  # Shape: (B, H, W)

        # 2. Format ground-truth mask to (B, H, W) float binary indicator
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        gt_foreground = (targets == 1).float()  # Shape: (B, H, W)

        # 3. Calculate Dice per-image (sum over spatial dimensions H and W)
        intersection = torch.sum(foreground_prob * gt_foreground, dim=(1, 2))  # (B,)
        cardinality = torch.sum(foreground_prob + gt_foreground, dim=(1, 2))    # (B,)

        dice_per_image = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)  # (B,)
        dice_loss_per_image = 1.0 - dice_per_image  # (B,)

        # 4. Average Dice loss across the batch
        return dice_loss_per_image.mean()


class CEDiceLoss(nn.Module):
    """
    Combined CrossEntropyLoss and Foreground-Only Soft Dice Loss.

    Loss Formulation:
        total_loss = alpha * ce_loss + beta * dice_loss

    Specification:
    - CrossEntropyLoss receives raw logits (B, 2, H, W) without prior softmax.
    - DiceLoss computes foreground (class 1) Dice per-image with smooth=1.0.
    - Default weights: alpha=1.0, beta=1.0.
    - Maintains logging for 'loss', 'ce', and 'dice'.
    - Fully agnostic to input channel count (3-7 channels).
    """

    def __init__(self, alpha=1.0, beta=1.0, smooth=1.0, weight=None, pos_weight=None):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pos_weight = float(pos_weight) if pos_weight is not None else 1.0
        self.smooth = float(smooth)
        if weight is None and pos_weight is not None and float(pos_weight) != 1.0:
            weight = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)
        elif weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits (torch.Tensor): Raw model logits of shape (B, 2, H, W)
            targets (torch.Tensor): Ground truth masks of shape (B, H, W) or (B, 1, H, W)
                                    with integer class indices {0, 1}

        Returns:
            total_loss (torch.Tensor): Combined loss for gradient backpropagation
            loss_dict (dict): Detached loss components for logging ('loss', 'ce', 'dice')
        """
        # Ensure targets are formatted as (B, H, W) long tensor for CrossEntropyLoss
        if targets.dim() == 4:
            targets_ce = targets.squeeze(1).long()
        else:
            targets_ce = targets.long()

        # 1. Pixel-wise CrossEntropyLoss on raw logits
        ce_loss = self.ce(logits, targets_ce)

        # 2. Foreground-only per-image Soft Dice Loss
        dice_loss = self.dice(logits, targets)

        # 3. Weighted combination
        total_loss = self.alpha * ce_loss + self.beta * dice_loss

        return total_loss, {
            'loss': total_loss.detach(),
            'ce': ce_loss.detach(),
            'dice': dice_loss.detach(),
            'bce': ce_loss.detach()  # Backward compatibility alias for logging
        }


# Aliases for seamless drop-in compatibility across the repository
BCEDiceLoss = CEDiceLoss
CombinedLoss = CEDiceLoss
