import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss supporting binary (1-channel or 2-channel) and multi-class segmentation.
    Computes smooth Dice overlap over batch predictions.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): raw unnormalized logits (B, C, H, W)
            targets (torch.Tensor): ground truth masks (B, 1, H, W) or (B, H, W) in {0, 1, ...}
        """
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1).float()
            intersection = (probs_flat * targets_flat).sum()
            cardinality = probs_flat.sum() + targets_flat.sum()
            dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1.0 - dice_score
        else:
            num_classes = logits.shape[1]
            probs = F.softmax(logits, dim=1)  # (B, C, H, W)

            if targets.dim() == 4 and targets.shape[1] == 1:
                targets_squeezed = targets.squeeze(1).long()
            else:
                targets_squeezed = targets.long()

            # One-hot encode targets -> (B, C, H, W)
            targets_one_hot = F.one_hot(targets_squeezed, num_classes=num_classes).permute(0, 3, 1, 2).float()

            dims = (0, 2, 3)
            intersection = torch.sum(probs * targets_one_hot, dims)
            cardinality = torch.sum(probs + targets_one_hot, dims)
            dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            return 1.0 - dice_per_class.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined CrossEntropy / BCEWithLogitsLoss and Soft Dice Loss.
    Supports both 1-channel (BCE) and multi-channel (CrossEntropy) segmentation.
    Loss = alpha * CE/BCE + beta * Dice
    """

    def __init__(self, alpha=1.0, beta=1.0, pos_weight=None, weight=None, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.weight = weight
        self.smooth = smooth
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Returns:
            total_loss (torch.Tensor): weighted sum of CE/BCE and Dice losses
            loss_components (dict): dictionary of individual loss values for logging
        """
        if logits.shape[1] == 1:
            targets_float = targets.float()
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, pos_weight=self.pos_weight)
            dice_loss = self.dice(logits, targets_float)
            total_loss = self.alpha * bce_loss + self.beta * dice_loss
            return total_loss, {
                'loss': total_loss.detach(),
                'bce': bce_loss.detach(),
                'dice': dice_loss.detach()
            }
        else:
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets_long = targets.squeeze(1).long()
            else:
                targets_long = targets.long()

            ce_loss = F.cross_entropy(logits, targets_long, weight=self.weight)
            dice_loss = self.dice(logits, targets)
            total_loss = self.alpha * ce_loss + self.beta * dice_loss
            return total_loss, {
                'loss': total_loss.detach(),
                'bce': ce_loss.detach(),  # alias for logger compatibility
                'ce': ce_loss.detach(),
                'dice': dice_loss.detach()
            }


# Alias for multi-class naming clarity
CEDiceLoss = BCEDiceLoss
