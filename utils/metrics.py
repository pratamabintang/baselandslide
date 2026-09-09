import torch
import torch.nn.functional as F
import numpy as np


class SegmentationMetrics:
    """
    Comprehensive accumulator and calculator for semantic segmentation metrics:
    - Global (Micro) Foreground IoU (Landslide IoU) & Background IoU
    - True Two-Class Mean IoU: mIoU = (Foreground IoU + Background IoU) / 2
    - Global (Micro) Foreground Dice (F1-Score) & Background Dice
    - True Two-Class Mean Dice: mDice = (Foreground Dice + Background Dice) / 2
    - Per-Image (Macro) Foreground Dice & IoU (equal tile weighting)
    - Per-Image Soft Dice (directly corresponding to 1.0 - DiceLoss)
    - Foreground Precision, Recall, and Overall Pixel Accuracy
    """

    def __init__(self, conf_thres=0.5, smooth=1e-7, loss_smooth=1.0):
        self.conf_thres = conf_thres
        self.smooth = smooth
        self.loss_smooth = loss_smooth
        self.reset()

    def reset(self):
        # Global micro accumulators
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.count = 0
        # Per-image macro metric accumulators
        self.image_fg_ious = []
        self.image_fg_dices = []
        self.image_soft_dices = []

    @torch.no_grad()
    def update(self, logits, targets):
        """
        Update running confusion matrix counts both globally and per-image.

        Args:
            logits (torch.Tensor): predicted logits (B, 2, H, W) or (B, 1, H, W)
            targets (torch.Tensor): ground truth binary masks (B, 1, H, W) or (B, H, W)
        """
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > self.conf_thres).float()
            fg_prob = probs
        else:
            probs = F.softmax(logits, dim=1)
            preds = (probs[:, 1:2, ...] > self.conf_thres).float()
            fg_prob = probs[:, 1:2, ...]

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = (targets > 0.5).float()

        batch_size = targets.size(0)

        # 1. Update Global (Micro) counts
        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()
        tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.count += batch_size

        # 2. Update Per-Image (Macro) metrics
        b_tp = (preds * targets).sum(dim=(1, 2, 3))
        b_fp = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
        b_fn = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))

        # Soft Dice terms (per image)
        b_soft_inter = (fg_prob * targets).sum(dim=(1, 2, 3))
        b_soft_card = (fg_prob + targets).sum(dim=(1, 2, 3))

        for i in range(batch_size):
            img_tp = b_tp[i].item()
            img_fp = b_fp[i].item()
            img_fn = b_fn[i].item()

            if img_tp + img_fp + img_fn == 0:
                img_iou = 1.0
                img_dice = 1.0
            else:
                img_iou = (img_tp + self.smooth) / (img_tp + img_fp + img_fn + self.smooth)
                img_dice = (2.0 * img_tp + self.smooth) / (2.0 * img_tp + img_fp + img_fn + self.smooth)

            soft_dice = (2.0 * b_soft_inter[i].item() + self.loss_smooth) / (b_soft_card[i].item() + self.loss_smooth)

            self.image_fg_ious.append(img_iou)
            self.image_fg_dices.append(img_dice)
            self.image_soft_dices.append(soft_dice)

    def compute(self):
        """
        Compute aggregate metrics over all accumulated samples.

        Returns:
            dict of float: Comprehensive segmentation metric suite
        """
        # Foreground (Landslide) Micro IoU & Dice
        if self.tp + self.fp + self.fn == 0:
            fg_iou = 1.0
            fg_dice = 1.0
        else:
            fg_iou = self.tp / (self.tp + self.fp + self.fn)
            fg_dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn)

        # Background Micro IoU & Dice
        if self.tn + self.fp + self.fn == 0:
            bg_iou = 1.0
            bg_dice = 1.0
        else:
            bg_iou = self.tn / (self.tn + self.fp + self.fn)
            bg_dice = (2.0 * self.tn) / (2.0 * self.tn + self.fp + self.fn)

        # True Two-Class Mean Metrics
        miou = (fg_iou + bg_iou) / 2.0
        mdice = (fg_dice + bg_dice) / 2.0

        # Foreground Precision & Recall
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        if self.tp + self.fn == 0:
            recall = 1.0 if (self.tp + self.fp == 0) else 0.0
        else:
            recall = self.tp / (self.tp + self.fn)

        # Overall Pixel Accuracy
        total_pixels = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / total_pixels if total_pixels > 0 else 0.0

        # Per-Image Macro Averages
        fg_dice_macro = float(np.mean(self.image_fg_dices)) if self.image_fg_dices else fg_dice
        fg_iou_macro = float(np.mean(self.image_fg_ious)) if self.image_fg_ious else fg_iou
        fg_dice_soft_macro = float(np.mean(self.image_soft_dices)) if self.image_soft_dices else fg_dice

        return {
            'iou': float(fg_iou),                 # Legacy alias for Foreground IoU (Micro)
            'fg_iou': float(fg_iou),              # Foreground (Landslide) IoU (Global / Micro)
            'bg_iou': float(bg_iou),              # Background IoU (Global / Micro)
            'miou': float(miou),                  # True Two-Class Mean IoU: (fg_iou + bg_iou) / 2
            'dice': float(fg_dice),               # Legacy alias for Foreground Dice (Micro)
            'fg_dice': float(fg_dice),            # Foreground (Landslide) Dice / F1 (Global / Micro)
            'bg_dice': float(bg_dice),            # Background Dice (Global / Micro)
            'mdice': float(mdice),                # True Two-Class Mean Dice: (fg_dice + bg_dice) / 2
            'fg_dice_macro': float(fg_dice_macro),# Per-image macro averaged binary Dice
            'fg_iou_macro': float(fg_iou_macro),  # Per-image macro averaged binary IoU
            'fg_dice_soft_macro': float(fg_dice_soft_macro), # Per-image soft macro Dice (matches 1.0 - DiceLoss)
            'precision': float(precision),        # Foreground Precision
            'recall': float(recall),              # Foreground Recall
            'accuracy': float(accuracy)           # Overall Pixel Accuracy
        }
