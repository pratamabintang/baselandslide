import os
import argparse
import glob
from pathlib import Path
import yaml
from PIL import Image
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.model_parser import Model
from data.datasets import resolve_inputs, get_channel_count, MODALITY_EXTENSIONS
from utils.torch_utils import select_device
from utils.general import increment_path, colorstr, check_file, set_logging

logger = set_logging(__name__)


def load_single_sample(source_stem, base_dir, inputs):
    """
    Load and preprocess multi-modal rasters for a single sample stem.

    Returns:
        tensor (torch.Tensor): (1, C_in, H, W)
        rgb_display (np.ndarray or None): (H, W, 3) image in [0, 1] for overlay plotting
    """
    base_dir = Path(base_dir)
    mod_arrays = []
    rgb_display = None

    for mod in inputs:
        mod_dir = base_dir / mod
        ext = MODALITY_EXTENSIONS.get(mod, '.tif')
        mod_path = mod_dir / f"{source_stem}{ext}"

        if not mod_path.exists():
            raise FileNotFoundError(f"Required modality file not found: {mod_path}")

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
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            rad = arr * (np.pi / 180.0)
            sin_aspect = np.sin(rad)
            cos_aspect = np.cos(rad)
            mod_arrays.append(np.stack([sin_aspect, cos_aspect], axis=-1))

        else:
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            mod_arrays.append(arr)

    combined = np.concatenate(mod_arrays, axis=-1)  # (H, W, C_in)
    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).float().unsqueeze(0)  # (1, C_in, H, W)

    return tensor, rgb_display


@torch.no_grad()
def run_predict(opt):
    device = select_device(opt.device)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)
    masks_dir = save_dir / 'masks'
    overlays_dir = save_dir / 'overlays'
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)

    # Load data YAML
    data_yaml = check_file(opt.data)
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    presets = data_dict.get('presets', {})

    # Load weights and configure model
    if opt.weights and Path(opt.weights).exists():
        ckpt = torch.load(opt.weights, map_location=device)
        model_cfg = ckpt.get('cfg', opt.cfg)
        inputs = ckpt.get('inputs', resolve_inputs(opt.inputs, presets))
        in_channels = ckpt.get('in_channels', get_channel_count(inputs, presets))
        nc = ckpt.get('nc', data_dict.get('nc', 1))

        model = Model(cfg=model_cfg, ch=in_channels, nc=nc).to(device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
        logger.info(f"Loaded model weights from {opt.weights}")
    else:
        inputs = resolve_inputs(opt.inputs, presets)
        in_channels = get_channel_count(inputs, presets)
        nc = data_dict.get('nc', 1)
        model = Model(cfg=opt.cfg, ch=in_channels, nc=nc).to(device)
        logger.info(f"Initialized unweighted model from {opt.cfg}")

    model.eval()

    # Discover target samples
    source_path = Path(opt.source)
    if source_path.is_file():
        base_dir = source_path.parent.parent
        stems = [source_path.stem]
    elif (source_path / 'IMAGE').exists():
        base_dir = source_path
        stems = [Path(f).stem for f in sorted(glob.glob(str(source_path / 'IMAGE' / '*.png')))]
    elif (source_path / 'DTM_NORM').exists():
        base_dir = source_path
        stems = [Path(f).stem for f in sorted(glob.glob(str(source_path / 'DTM_NORM' / '*.tif')))]
    else:
        # Scan current dir
        base_dir = source_path
        files = glob.glob(str(source_path / '**' / '*.png'), recursive=True)
        stems = list(dict.fromkeys([Path(f).stem for f in sorted(files)]))

    if len(stems) == 0:
        logger.warning(f"No samples found at {opt.source}")
        return

    logger.info(f"Running inference on {len(stems)} samples...")

    for stem in stems:
        try:
            tensor, rgb_display = load_single_sample(stem, base_dir, inputs)
            tensor = tensor.to(device)

            logits = model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)
            binary_mask = (probs > opt.conf_thres).astype(np.uint8) * 255

            # Save binary PNG mask
            mask_out_path = masks_dir / f"{stem}.png"
            Image.fromarray(binary_mask).save(mask_out_path)

            # Save visual overlay
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            if rgb_display is not None:
                ax[0].imshow(rgb_display)
                ax[0].set_title("Input RGB")
            else:
                ax[0].imshow(tensor[0, 0].cpu().numpy(), cmap='terrain')
                ax[0].set_title("Input Topography")
            ax[0].axis('off')

            ax[1].imshow(probs, cmap='jet', vmin=0, vmax=1)
            ax[1].set_title("Probability Heatmap")
            ax[1].axis('off')

            if rgb_display is not None:
                overlay = rgb_display.copy()
                mask_bool = binary_mask > 0
                overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 0.5, 0, 1)  # Red highlight
                ax[2].imshow(overlay)
            else:
                ax[2].imshow(binary_mask, cmap='gray')
            ax[2].set_title(f"Prediction (thr={opt.conf_thres})")
            ax[2].axis('off')

            plt.tight_layout()
            overlay_out_path = overlays_dir / f"{stem}_overlay.png"
            plt.savefig(overlay_out_path, dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Error processing sample {stem}: {e}")

    print("\n" + "=" * 70)
    print(colorstr('bold', 'green', '[DONE] INFERENCE COMPLETE'))
    print("=" * 70)
    print(f"  Processed Samples : {len(stems)}")
    print(f"  Saved Masks       : {masks_dir}")
    print(f"  Saved Overlays    : {overlays_dir}")
    print("=" * 70 + "\n")


def parse_opt():
    parser = argparse.ArgumentParser(description="Run Landslide Segmentation Inference")
    parser.add_argument('--weights', type=str, default='', help='model weights (.pt)')
    parser.add_argument('--source', type=str, default='dataset/dataset_1/validation', help='input sample or folder')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--inputs', type=str, default='rgb_only',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold for segmentation mask')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/predict', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    run_predict(opt)
