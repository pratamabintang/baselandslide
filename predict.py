import os
import sys
import time
import argparse
import glob
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.model_parser import Model
from data.datasets import resolve_inputs, get_channel_count, MODALITY_EXTENSIONS
from utils.torch_utils import select_device, torch_load
from utils.general import increment_path, colorstr, check_file, set_logging, parse_options_with_config

logger = set_logging(__name__)


def load_single_sample(source_stem, base_dir, inputs, label_folder='LABEL', fusion='concat', img_size=None):
    """
    Load and preprocess multi-modal rasters for a single sample stem,
    and optionally load ground truth label mask if present.

    Args:
        source_stem (str): Sample identifier (e.g. 'Chainage_17_00156')
        base_dir (Path): Base split or dataset directory
        inputs (list): List of active modality names (e.g. ['IMAGE', 'DTM_NORM'])
        label_folder (str): Name of ground truth mask folder
        fusion (str): Modality fusion mode ('concat' or 'add')
        img_size (int, tuple, or None): Optional target spatial resolution (H, W)

    Returns:
        tensor (torch.Tensor): Model input tensor of shape (1, C_in, H, W)
        rgb_display (np.ndarray or None): (H, W, 3) normalized RGB image in [0, 1]
        gt_mask (np.ndarray or None): (H, W) binary float array in {0.0, 1.0} or None if not found
    """
    base_dir = Path(base_dir)
    mod_arrays = []
    rgb_display = None

    # 1. Load active input modalities
    for mod in inputs:
        mod_dir = base_dir / mod
        ext = MODALITY_EXTENSIONS.get(mod, '.tif')
        mod_path = mod_dir / f"{source_stem}{ext}"

        # Fallback search if exact path not in standard folder
        if not mod_path.exists():
            matched_files = list(base_dir.glob(f"**/{mod}/{source_stem}.*"))
            if matched_files:
                mod_path = matched_files[0]
            else:
                raise FileNotFoundError(f"Required modality '{mod}' file not found for sample '{source_stem}' in {base_dir}")

        if mod == 'IMAGE':
            img = Image.open(mod_path).convert('RGB')
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / 255.0
            rgb_display = arr.copy()
            mod_arrays.append(arr)

        elif mod == 'DTM_NORM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / 65535.0
            mod_arrays.append(np.expand_dims(arr, axis=-1))

        elif mod == 'DTM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            d_min, d_max = -2.40497088432312, 170.77879333496094
            arr = np.clip((arr - d_min) / (d_max - d_min + 1e-7), 0.0, 1.0)
            mod_arrays.append(np.expand_dims(arr, axis=-1))

        elif mod == 'SLOPE':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr / 90.0, 0.0, 1.0)
            mod_arrays.append(np.expand_dims(arr, axis=-1))

        elif mod == 'ASPECT':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            invalid_mask = np.isnan(arr) | np.isinf(arr) | (arr < 0) | (arr > 360.0)
            valid_arr = np.where(invalid_mask, 0.0, arr)
            rad = valid_arr * (np.pi / 180.0)
            sin_aspect = np.sin(rad)
            cos_aspect = np.cos(rad)
            sin_aspect[invalid_mask] = 0.0
            cos_aspect[invalid_mask] = 0.0
            mod_arrays.append(np.stack([sin_aspect, cos_aspect], axis=-1))

        else:
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            mod_arrays.append(arr)

    if str(fusion).lower() in ['add', 'addition', 'sum']:
        if 'IMAGE' in inputs:
            rgb_idx = inputs.index('IMAGE')
            fused = mod_arrays[rgb_idx].copy()
            aux_count = 0
            for i, mod in enumerate(inputs):
                if i != rgb_idx:
                    aux = mod_arrays[i]
                    if aux.ndim == 3 and aux.shape[-1] != 1:
                        raise ValueError(
                            f"Additive fusion ('add') cannot add multi-channel modality '{mod}' (shape {aux.shape}) "
                            f"to RGB base. Only 1-channel scalar modalities (e.g. DTM_NORM, SLOPE) can be added."
                        )
                    fused = fused + aux
                    aux_count += 1
            combined = fused / (1.0 + aux_count)
        else:
            fused = mod_arrays[0].copy()
            for aux in mod_arrays[1:]:
                if aux.ndim == 3 and aux.shape[-1] != 1:
                    raise ValueError(
                        f"Additive fusion ('add') cannot add multi-channel modality (shape {aux.shape}) to base."
                    )
                fused = fused + aux
            combined = fused / float(len(mod_arrays))
    else:
        combined = np.concatenate(mod_arrays, axis=-1)  # (H, W, C_in)

    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0)  # (1, C_in, H, W)

    # 2. Check and load Ground Truth Label Mask if available
    gt_mask = None
    label_candidates = [
        base_dir / label_folder / f"{source_stem}.png",
        base_dir / label_folder / f"{source_stem}.tif",
        base_dir.parent / label_folder / f"{source_stem}.png",
    ]
    for candidate in label_candidates:
        if candidate.exists():
            img_gt = Image.open(candidate)
            arr_gt = np.array(img_gt)
            gt_mask = (arr_gt > 0).astype(np.float32)
            break

    # 3. Optional spatial dimension enforcement
    if img_size is not None:
        target_h, target_w = (img_size, img_size) if isinstance(img_size, int) else img_size
        if tensor.shape[2] != target_h or tensor.shape[3] != target_w:
            tensor = torch.nn.functional.interpolate(tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
            if gt_mask is not None:
                gt_t = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float()
                gt_mask = torch.nn.functional.interpolate(gt_t, size=(target_h, target_w), mode='nearest')[0, 0].numpy()
            if rgb_display is not None:
                rgb_t = torch.from_numpy(rgb_display.transpose(2, 0, 1)).unsqueeze(0).float()
                rgb_display = torch.nn.functional.interpolate(rgb_t, size=(target_h, target_w), mode='bilinear', align_corners=False)[0].numpy().transpose(1, 2, 0)

    return tensor, rgb_display, gt_mask


def compute_sample_metrics(pred_mask, gt_mask):
    """
    Compute binary and two-class segmentation metrics for a single sample.

    Args:
        pred_mask (np.ndarray): Binary prediction {0, 1}
        gt_mask (np.ndarray): Binary ground truth {0, 1}

    Returns:
        dict: fg_iou, bg_iou, miou, fg_dice, bg_dice, mdice, precision, recall, tp, fp, fn, tn
    """
    pred_b = (pred_mask > 0).astype(bool)
    gt_b = (gt_mask > 0).astype(bool)

    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
    tn = np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum()

    # Foreground IoU & Dice (Landslide)
    union_fg = tp + fp + fn
    fg_iou = 1.0 if union_fg == 0 else float(tp) / float(union_fg)
    card_fg = 2 * tp + fp + fn
    fg_dice = 1.0 if card_fg == 0 else float(2 * tp) / float(card_fg)

    # Background IoU & Dice
    union_bg = tn + fp + fn
    bg_iou = 1.0 if union_bg == 0 else float(tn) / float(union_bg)
    card_bg = 2 * tn + fp + fn
    bg_dice = 1.0 if card_bg == 0 else float(2 * tn) / float(card_bg)

    # Two-Class Means
    miou = (fg_iou + bg_iou) / 2.0
    mdice = (fg_dice + bg_dice) / 2.0

    # Precision & Recall
    if tp + fp == 0:
        precision = 1.0 if gt_b.sum() == 0 else 0.0
    else:
        precision = float(tp) / float(tp + fp)

    if tp + fn == 0:
        recall = 1.0 if (tp + fp == 0) else 0.0
    else:
        recall = float(tp) / float(tp + fn)

    return {
        'iou': float(fg_iou),
        'fg_iou': float(fg_iou),
        'bg_iou': float(bg_iou),
        'miou': float(miou),
        'dice': float(fg_dice),
        'fg_dice': float(fg_dice),
        'bg_dice': float(bg_dice),
        'mdice': float(mdice),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'gt_pixels': int(gt_b.sum()),
        'pred_pixels': int(pred_b.sum())
    }


def create_comparison_figure(stem, rgb_display, tensor, probs, pred_mask, gt_mask, conf_thres=0.5):
    """
    Generate a publication-grade comparison figure comparing model prediction with Ground Truth.

    If Ground Truth is available: 5-Panel Layout
      [1] Input Image | [2] Ground Truth Overlay | [3] Predicted Overlay | [4] Confidence Heatmap | [5] Confusion Map (TP/FP/FN)
    If Ground Truth is NOT available: 3-Panel Layout
      [1] Input Image | [2] Confidence Heatmap | [3] Predicted Overlay
    """
    h, w = probs.shape
    has_gt = gt_mask is not None

    if has_gt:
        metrics = compute_sample_metrics(pred_mask, gt_mask)
        fig, ax = plt.subplots(1, 5, figsize=(25, 5), dpi=150)

        # Base backdrop for overlays (RGB or grayscale Topography)
        if rgb_display is not None:
            base_bg = rgb_display.copy()
        else:
            topo = tensor[0, 0].cpu().numpy()
            topo_norm = (topo - topo.min()) / (topo.max() - topo.min() + 1e-7)
            base_bg = np.stack([topo_norm] * 3, axis=-1)

        # -------------------------------------------------------------
        # Panel 1: Input (RGB / Terrain)
        # -------------------------------------------------------------
        ax[0].imshow(base_bg)
        ax[0].set_title(f"Input: {stem}", fontsize=12, fontweight='bold')
        ax[0].axis('off')

        # -------------------------------------------------------------
        # Panel 2: Ground Truth Overlay (Lime Green #39FF14)
        # -------------------------------------------------------------
        gt_overlay = base_bg.copy()
        gt_bool = gt_mask > 0
        if gt_bool.any():
            # Blend 50% image + 50% green
            gt_overlay[gt_bool] = np.clip(gt_overlay[gt_bool] * 0.4 + np.array([0.0, 0.95, 0.1]) * 0.6, 0, 1)
        ax[1].imshow(gt_overlay)
        ax[1].contour(gt_mask, levels=[0.5], colors=['#00FF00'], linewidths=1.2)
        gt_pct = (gt_bool.sum() / (h * w)) * 100
        ax[1].set_title(f"Ground Truth (GT: {gt_pct:.2f}%)", fontsize=12, fontweight='bold', color='darkgreen')
        ax[1].axis('off')

        # -------------------------------------------------------------
        # Panel 3: Model Prediction Overlay (Bright Red #FF2A2A)
        # -------------------------------------------------------------
        pred_overlay = base_bg.copy()
        pred_bool = pred_mask > 0
        if pred_bool.any():
            # Blend 50% image + 50% red
            pred_overlay[pred_bool] = np.clip(pred_overlay[pred_bool] * 0.4 + np.array([1.0, 0.15, 0.15]) * 0.6, 0, 1)
        ax[2].imshow(pred_overlay)
        if pred_bool.any():
            ax[2].contour(pred_mask, levels=[0.5], colors=['#FF0000'], linewidths=1.2)
        pred_pct = (pred_bool.sum() / (h * w)) * 100
        ax[2].set_title(f"Prediction (thr={conf_thres:.2f}, Area: {pred_pct:.2f}%)", fontsize=12, fontweight='bold', color='darkred')
        ax[2].axis('off')

        # -------------------------------------------------------------
        # Panel 4: Confidence Probability Heatmap
        # -------------------------------------------------------------
        im_heat = ax[3].imshow(probs, cmap='jet', vmin=0.0, vmax=1.0)
        ax[3].set_title(f"Probability Heatmap (max={probs.max():.2f})", fontsize=12, fontweight='bold')
        ax[3].axis('off')
        plt.colorbar(im_heat, ax=ax[3], fraction=0.046, pad=0.04)

        # -------------------------------------------------------------
        # Panel 5: Confusion Error Map (TP=Green, FP=Red, FN=Cyan)
        # -------------------------------------------------------------
        confusion_img = base_bg.copy()
        tp_b = np.logical_and(pred_bool, gt_bool)
        fp_b = np.logical_and(pred_bool, np.logical_not(gt_bool))
        fn_b = np.logical_and(np.logical_not(pred_bool), gt_bool)

        confusion_img[tp_b] = np.clip(confusion_img[tp_b] * 0.3 + np.array([0.0, 1.0, 0.0]) * 0.7, 0, 1)    # Green: True Positive
        confusion_img[fp_b] = np.clip(confusion_img[fp_b] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7, 0, 1)    # Red: False Positive (Over-pred)
        confusion_img[fn_b] = np.clip(confusion_img[fn_b] * 0.3 + np.array([0.0, 0.6, 1.0]) * 0.7, 0, 1)    # Cyan: False Negative (Miss)

        ax[4].imshow(confusion_img)
        ax[4].set_title(f"Confusion: Fg IoU={metrics['fg_iou']*100:.1f}%, mIoU={metrics['miou']*100:.1f}%, Fg Dice={metrics['fg_dice']*100:.1f}%", fontsize=11, fontweight='bold')
        ax[4].axis('off')

        # Add Legend
        patches = [
            mpatches.Patch(color='#00FF00', label='TP (Hit)'),
            mpatches.Patch(color='#FF0000', label='FP (False Alarm)'),
            mpatches.Patch(color='#0099FF', label='FN (Miss)')
        ]
        ax[4].legend(handles=patches, loc='lower right', framealpha=0.85, fontsize=8)

        plt.tight_layout()
        return fig, metrics

    else:
        # 3-Panel Layout when no Ground Truth is available
        fig, ax = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
        if rgb_display is not None:
            base_bg = rgb_display
        else:
            topo = tensor[0, 0].cpu().numpy()
            base_bg = (topo - topo.min()) / (topo.max() - topo.min() + 1e-7)

        # Panel 1: Input
        ax[0].imshow(base_bg)
        ax[0].set_title(f"Input: {stem}", fontsize=12, fontweight='bold')
        ax[0].axis('off')

        # Panel 2: Heatmap
        im_heat = ax[1].imshow(probs, cmap='jet', vmin=0.0, vmax=1.0)
        ax[1].set_title(f"Confidence Heatmap (max={probs.max():.2f})", fontsize=12, fontweight='bold')
        ax[1].axis('off')
        plt.colorbar(im_heat, ax=ax[1], fraction=0.046, pad=0.04)

        # Panel 3: Prediction Overlay
        pred_overlay = base_bg.copy() if base_bg.ndim == 3 else np.stack([base_bg]*3, axis=-1)
        pred_bool = pred_mask > 0
        if pred_bool.any():
            pred_overlay[pred_bool] = np.clip(pred_overlay[pred_bool] * 0.4 + np.array([1.0, 0.2, 0.2]) * 0.6, 0, 1)
        ax[2].imshow(pred_overlay)
        if pred_bool.any():
            ax[2].contour(pred_mask, levels=[0.5], colors=['#FF0000'], linewidths=1.2)
        ax[2].set_title(f"Predicted Overlay (thr={conf_thres:.2f})", fontsize=12, fontweight='bold')
        ax[2].axis('off')

        plt.tight_layout()
        return fig, None


@torch.no_grad()
def run_predict(opt):
    device = select_device(opt.device)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)
    masks_dir = save_dir / 'masks'
    overlays_dir = save_dir / 'overlays'
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    # Load data YAML configuration
    data_yaml = check_file(opt.data)
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    presets = data_dict.get('presets', {})
    fusion = getattr(opt, 'fusion', 'concat')

    # Load model and weights
    if opt.weights and Path(opt.weights).exists():
        ckpt = torch_load(opt.weights, map_location=device)
        model_cfg = ckpt.get('cfg', opt.cfg)
        fusion = ckpt.get('fusion', fusion)
        inputs = ckpt.get('inputs', resolve_inputs(opt.inputs, presets))
        in_channels = ckpt.get('in_channels', get_channel_count(inputs, presets, fusion=fusion))
        nc = ckpt.get('nc', data_dict.get('nc', 1))

        model = Model(cfg=model_cfg, ch=in_channels, nc=nc).to(device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        ep_info = f" (Epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and 'epoch' in ckpt else ""
        logger.info(f"Loaded model weights from {opt.weights}{ep_info}")
    else:
        inputs = resolve_inputs(opt.inputs, presets)
        in_channels = get_channel_count(inputs, presets, fusion=fusion)
        nc = data_dict.get('nc', 1)
        model = Model(cfg=opt.cfg, ch=in_channels, nc=nc).to(device)
        logger.info(f"Initialized unweighted model from {opt.cfg}")

    model.eval()

    # Discover target samples
    source_path = Path(opt.source)
    if source_path.is_file():
        base_dir = source_path.parent.parent if source_path.parent.name in ['IMAGE', 'DTM_NORM', 'LABEL'] else source_path.parent
        stems = [source_path.stem]
    elif (source_path / 'IMAGE').exists():
        base_dir = source_path
        stems = [Path(f).stem for f in sorted(glob.glob(str(source_path / 'IMAGE' / '*.png')))]
    elif (source_path / 'DTM_NORM').exists():
        base_dir = source_path
        stems = [Path(f).stem for f in sorted(glob.glob(str(source_path / 'DTM_NORM' / '*.tif')))]
    else:
        base_dir = source_path
        files = glob.glob(str(source_path / '**' / '*.png'), recursive=True)
        stems = list(dict.fromkeys([Path(f).stem for f in sorted(files)]))

    if len(stems) == 0:
        logger.warning(f"No samples found at source: '{opt.source}'")
        return

    print("\n" + "=" * 80)
    print(colorstr('bold', 'cyan', f"[PREDICT] RUNNING INFERENCE & GROUND TRUTH OVERLAY COMPARISON"))
    print("=" * 80)
    print(f"  Source Path      : {opt.source}")
    print(f"  Total Samples    : {len(stems)}")
    print(f"  Input Modalities : {inputs} (Channels: {in_channels}, Fusion: {fusion})")
    print(f"  Binary Threshold : {opt.conf_thres}")
    print(f"  Save Directory   : {save_dir}")
    print("-" * 80)

    pbar = tqdm(stems, desc="Predicting", unit="img", bar_format="{l_bar}{bar:25}{r_bar}")
    t_infer_start = time.time()
    processed_count = 0
    all_sample_metrics = []

    for stem in pbar:
        try:
            tensor, rgb_display, gt_mask = load_single_sample(stem, base_dir, inputs, fusion=fusion)
            tensor = tensor.to(device)

            # Forward pass
            cuda = device.type == 'cuda'
            if cuda:
                with torch.amp.autocast('cuda'):
                    logits = model(tensor)
            else:
                logits = model(tensor)

            # Probability extraction
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

            binary_mask = (probs > opt.conf_thres).astype(np.uint8)

            # 1. Save binary PNG mask
            mask_png_path = masks_dir / f"{stem}.png"
            Image.fromarray(binary_mask * 255).save(mask_png_path)

            # 2. Save 5-Panel (or 3-Panel) Comparison Figure with Ground Truth
            fig, metrics = create_comparison_figure(
                stem=stem,
                rgb_display=rgb_display,
                tensor=tensor,
                probs=probs,
                pred_mask=binary_mask,
                gt_mask=gt_mask,
                conf_thres=opt.conf_thres
            )
            overlay_png_path = overlays_dir / f"{stem}_overlay.png"
            fig.savefig(overlay_png_path, bbox_inches='tight')
            plt.close(fig)

            if metrics:
                metrics['stem'] = stem
                all_sample_metrics.append(metrics)

            processed_count += 1
            elapsed = time.time() - t_infer_start
            fps = processed_count / max(elapsed, 1e-4)
            pbar.set_postfix({'speed': f"{fps:.1f} img/s", 'last': stem[:16]})

        except Exception as e:
            logger.error(f"Error processing sample {stem}: {e}")

    total_infer_time = time.time() - t_infer_start

    # Print Summary & Save CSV if Ground Truth was available
    print("\n" + "=" * 80)
    print(colorstr('bold', 'green', '[DONE] INFERENCE & VISUALIZATION COMPLETE'))
    print("=" * 80)
    print(f"  Processed Samples : {processed_count}/{len(stems)} in {total_infer_time:.2f}s ({processed_count/max(total_infer_time,1e-4):.1f} img/s)")
    print(f"  Saved Masks       : {masks_dir}")
    print(f"  Saved Overlays    : {overlays_dir}")

    if all_sample_metrics:
        # Save per-sample metrics CSV
        csv_path = save_dir / 'summary.csv'
        with open(csv_path, 'w') as f:
            f.write("stem,fg_iou,bg_iou,miou,fg_dice,bg_dice,mdice,precision,recall,tp,fp,fn,tn,gt_pixels,pred_pixels\n")
            for m in all_sample_metrics:
                f.write(f"{m['stem']},{m['fg_iou']:.5f},{m['bg_iou']:.5f},{m['miou']:.5f},{m['fg_dice']:.5f},{m['bg_dice']:.5f},{m['mdice']:.5f},{m['precision']:.5f},{m['recall']:.5f},"
                        f"{m['tp']},{m['fp']},{m['fn']},{m['tn']},{m['gt_pixels']},{m['pred_pixels']}\n")

        # Compute dataset averages
        mean_fg_iou = np.mean([m['fg_iou'] for m in all_sample_metrics])
        mean_bg_iou = np.mean([m['bg_iou'] for m in all_sample_metrics])
        mean_miou = np.mean([m['miou'] for m in all_sample_metrics])
        mean_fg_dice = np.mean([m['fg_dice'] for m in all_sample_metrics])
        mean_bg_dice = np.mean([m['bg_dice'] for m in all_sample_metrics])
        mean_mdice = np.mean([m['mdice'] for m in all_sample_metrics])
        mean_prec = np.mean([m['precision'] for m in all_sample_metrics])
        mean_rec = np.mean([m['recall'] for m in all_sample_metrics])

        print("-" * 80)
        print(colorstr('bold', 'yellow', f"GROUND TRUTH EVALUATION SUMMARY ({len(all_sample_metrics)} Ground Truth Samples):"))
        print(f"  Mean Foreground IoU (Landslide) : {mean_fg_iou * 100:6.2f}%")
        print(f"  Mean Two-Class mIoU             : {mean_miou * 100:6.2f}%")
        print(f"  Mean Background IoU             : {mean_bg_iou * 100:6.2f}%")
        print(f"  Mean Foreground Dice (F1)       : {mean_fg_dice * 100:6.2f}%")
        print(f"  Mean Two-Class mDice            : {mean_mdice * 100:6.2f}%")
        print(f"  Mean Precision                  : {mean_prec * 100:6.2f}%")
        print(f"  Mean Recall                     : {mean_rec * 100:6.2f}%")
        print(f"  Detailed CSV                    : {csv_path}")

    print("=" * 80 + "\n")


def parse_opt():
    parser = argparse.ArgumentParser(description="Run Landslide Segmentation Inference & Ground Truth Comparison")
    parser.add_argument('--config', type=str, default='', help='path to yaml configuration file (e.g. configs/predict.yaml)')
    parser.add_argument('--weights', type=str, default='', help='model weights (.pt)')
    parser.add_argument('--source', type=str, default='dataset/dataset_1/validation', help='input sample file or directory')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--inputs', type=str, default='rgb_dtm',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all, rgb_add_dtm)')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add'],
                        help='modality fusion mode: concat (channel concatenation) or add (element-wise addition into RGB)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold for binary segmentation mask')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/predict', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parse_options_with_config(parser)


if __name__ == '__main__':
    opt = parse_opt()
    run_predict(opt)
