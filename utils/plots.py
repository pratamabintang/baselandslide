import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def plot_results(csv_path, save_dir=''):
    """
    Plot training metrics and loss curves from results.csv.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    try:
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data[None, :]

        epochs = data[:, 0]
        train_loss = data[:, 1]
        val_loss = data[:, 2]
        val_iou = data[:, 3]
        val_dice = data[:, 4]
        val_prec = data[:, 5]
        val_rec = data[:, 6]

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # Best metrics indices
        best_dice_idx = int(np.argmax(val_dice))
        best_loss_idx = int(np.argmin(val_loss))
        best_iou_idx = int(np.argmax(val_iou))
        best_dice_epoch = int(epochs[best_dice_idx])

        # 1. Loss Curve
        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss', alpha=0.85)
        axes[0, 0].plot(epochs, val_loss, 'r--', label='Val Loss', alpha=0.85)
        axes[0, 0].scatter(epochs[best_loss_idx], val_loss[best_loss_idx], color='red', s=90, zorder=5, edgecolors='black',
                           label=f'★ Min Val: {val_loss[best_loss_idx]:.4f} (Ep {int(epochs[best_loss_idx])})')
        axes[0, 0].set_title('Loss vs Epochs', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, linestyle=':', alpha=0.6)
        axes[0, 0].legend(loc='best')

        # 2. IoU Curve
        axes[0, 1].plot(epochs, val_iou * 100, 'g-', label='Val mIoU', alpha=0.85)
        axes[0, 1].scatter(epochs[best_iou_idx], val_iou[best_iou_idx] * 100, color='forestgreen', s=90, zorder=5, edgecolors='black',
                           label=f'★ Best mIoU: {val_iou[best_iou_idx]*100:.2f}% (Ep {int(epochs[best_iou_idx])})')
        axes[0, 1].set_title('Mean IoU (Jaccard Index)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('mIoU (%)')
        axes[0, 1].grid(True, linestyle=':', alpha=0.6)
        axes[0, 1].legend(loc='best')

        # 3. Dice / F1 Score
        axes[1, 0].plot(epochs, val_dice * 100, 'm-', label='Val Dice / F1', alpha=0.85)
        axes[1, 0].scatter(epochs[best_dice_idx], val_dice[best_dice_idx] * 100, color='gold', marker='*', s=220, zorder=5, edgecolors='black',
                           label=f'★ Best Dice: {val_dice[best_dice_idx]*100:.2f}% (Ep {best_dice_epoch})')
        axes[1, 0].set_title('Dice Coefficient (F1-Score)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice (%)')
        axes[1, 0].grid(True, linestyle=':', alpha=0.6)
        axes[1, 0].legend(loc='best')

        # 4. Precision & Recall
        axes[1, 1].plot(epochs, val_prec * 100, 'c-', label='Precision', alpha=0.85)
        axes[1, 1].plot(epochs, val_rec * 100, 'darkorange', linestyle='--', label='Recall', alpha=0.85)
        axes[1, 1].scatter(epochs[best_dice_idx], val_prec[best_dice_idx] * 100, color='cyan', s=70, zorder=5, edgecolors='black',
                           label=f'Prec @ Best: {val_prec[best_dice_idx]*100:.2f}%')
        axes[1, 1].scatter(epochs[best_dice_idx], val_rec[best_dice_idx] * 100, color='darkorange', s=70, zorder=5, edgecolors='black',
                           label=f'Rec @ Best: {val_rec[best_dice_idx]*100:.2f}%')
        axes[1, 1].set_title('Precision & Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score (%)')
        axes[1, 1].grid(True, linestyle=':', alpha=0.6)
        axes[1, 1].legend(loc='best')

        plt.tight_layout()
        save_path = Path(save_dir) / 'results.png' if save_dir else csv_path.parent / 'results.png'
        plt.savefig(save_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[Warning] Failed to plot results: {e}")


def plot_predictions(tensors, targets, preds_logits, save_path, conf_thres=0.5, max_samples=4):
    """
    Generate comparison plot of inputs, ground-truth masks, predicted heatmaps, and binary overlay.

    Args:
        tensors (torch.Tensor): multi-modal input tensors (B, C, H, W)
        targets (torch.Tensor): ground truth masks (B, 1, H, W)
        preds_logits (torch.Tensor): predicted logits (B, 1, H, W)
        save_path (str or Path): output image destination
    """
    n = min(tensors.size(0), max_samples)
    if preds_logits.shape[1] == 1:
        probs = torch.sigmoid(preds_logits)
    else:
        probs = torch.softmax(preds_logits, dim=1)[:, 1:2, ...]
    binary_preds = (probs > conf_thres).float()

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        # Base image: if 3+ channels and optical RGB available, show first 3 channels
        img_np = tensors[i].detach().cpu().numpy()
        if img_np.shape[0] >= 3:
            base_display = np.clip(img_np[:3].transpose(1, 2, 0), 0.0, 1.0)
            cmap_base = None
        else:
            base_display = img_np[0]
            cmap_base = 'terrain'

        target_np = targets[i, 0].detach().cpu().numpy()
        prob_np = probs[i, 0].detach().cpu().numpy()
        bin_np = binary_preds[i, 0].detach().cpu().numpy()

        # 1. Base input
        axes[i, 0].imshow(base_display, cmap=cmap_base)
        axes[i, 0].set_title(f"Sample {i+1}: Input")
        axes[i, 0].axis('off')

        # 2. Ground Truth
        axes[i, 1].imshow(target_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis('off')

        # 3. Probability Heatmap
        im = axes[i, 2].imshow(prob_np, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title("Predicted Probability")
        axes[i, 2].axis('off')

        # 4. Binary Decision Overlay
        axes[i, 3].imshow(bin_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title(f"Binary Prediction (thr={conf_thres})")
        axes[i, 3].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
