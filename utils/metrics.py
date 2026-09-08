import torch
import torch.nn.functional as F
import numpy as np


class SegmentationMetrics:
    """
    Accumulator and calculator for binary segmentation metrics:
    - Intersection-over-Union (IoU / Jaccard Index)
    - Dice Coefficient (F1-Score)
    - Precision
    - Recall
    - Pixel Accuracy
    """

    def __init__(self, conf_thres=0.5, smooth=1e-7):
        self.conf_thres = conf_thres
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, logits, targets):
        """
        Update running confusion matrix counts.

        Args:
            logits (torch.Tensor): predicted logits (B, 1, H, W)
            targets (torch.Tensor): ground truth binary masks (B, 1, H, W)
        """
        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)
            preds = (probs > self.conf_thres).float()
        else:
            probs = F.softmax(logits, dim=1)
            preds = (probs[:, 1:2, ...] > self.conf_thres).float()

        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = (targets > 0.5).float()

        # Compute batch elements
        tp = (preds * targets).sum().item()
        fp = (preds * (1.0 - targets)).sum().item()
        fn = ((1.0 - preds) * targets).sum().item()
        tn = ((1.0 - preds) * (1.0 - targets)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        self.count += targets.size(0)

    def compute(self):
        """
        Compute aggregate metrics over all accumulated samples.

        Returns:
            dict of float: iou, dice, precision, recall, accuracy
        """
        iou = (self.tp + self.smooth) / (self.tp + self.fp + self.fn + self.smooth)
        dice = (2.0 * self.tp + self.smooth) / (2.0 * self.tp + self.fp + self.fn + self.smooth)
        precision = (self.tp + self.smooth) / (self.tp + self.fp + self.smooth)
        recall = (self.tp + self.smooth) / (self.tp + self.fn + self.smooth)
        accuracy = (self.tp + self.tn + self.smooth) / (self.tp + self.tn + self.fp + self.fn + self.smooth)

        return {
            'iou': float(iou),
            'dice': float(dice),
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy)
        }
