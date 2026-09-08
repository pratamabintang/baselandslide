import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Computes smooth Dice overlap over batch predictions.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): raw unnormalized logits (B, 1, H, W)
            targets (torch.Tensor): ground truth binary masks (B, 1, H, W) in {0, 1}
        """
        probs = torch.sigmoid(logits)

        # Flatten batch and spatial dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and Soft Dice Loss.
    Loss = alpha * BCE + beta * Dice
    """

    def __init__(self, alpha=1.0, beta=1.0, pos_weight=None, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Returns:
            total_loss (torch.Tensor): weighted sum of BCE and Dice losses
            loss_components (dict): dictionary of individual loss values for logging
        """
        targets = targets.float()
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.alpha * bce_loss + self.beta * dice_loss

        return total_loss, {
            'loss': total_loss.detach(),
            'bce': bce_loss.detach(),
            'dice': dice_loss.detach()
        }
