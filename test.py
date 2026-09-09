import argparse
import sys
from pathlib import Path
import yaml
import torch
from tqdm import tqdm

from models.model_parser import Model
from data.datasets import create_dataloader, resolve_inputs, get_channel_count
from utils.loss import BCEDiceLoss
from utils.metrics import SegmentationMetrics
from utils.plots import plot_predictions
from utils.torch_utils import select_device, torch_load
from utils.general import increment_path, colorstr, check_file, set_logging, parse_options_with_config

logger = set_logging(__name__)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    conf_thres=0.5,
    save_dir=None,
    plots=True,
    pbar_desc="Validating"
):
    """
    Run evaluation loop over dataset, compute losses and segmentation metrics.

    Returns:
        metrics_dict (dict): loss, iou, dice, precision, recall, accuracy
    """
    model.eval()
    metrics = SegmentationMetrics(conf_thres=conf_thres)

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=pbar_desc, leave=False, total=num_batches)
    saved_plot = False

    cuda = device.type == 'cuda'
    for batch_idx, (tensors, targets, stems) in enumerate(pbar):
        tensors = tensors.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cuda:
            with torch.amp.autocast('cuda'):
                logits = model(tensors)
                loss, loss_dict = criterion(logits, targets)
        else:
            logits = model(tensors)
            loss, loss_dict = criterion(logits, targets)

        total_loss += loss_dict['loss'].item()
        total_bce += loss_dict['bce'].item()
        total_dice += loss_dict['dice'].item()

        metrics.update(logits, targets)


        # Save first batch visualization if requested
        if plots and save_dir and not saved_plot:
            plot_path = Path(save_dir) / 'val_predictions.png'
            plot_predictions(tensors, targets, logits, plot_path, conf_thres=conf_thres)
            saved_plot = True

    avg_loss = total_loss / max(num_batches, 1)
    avg_bce = total_bce / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)

    scores = metrics.compute()
    scores['loss'] = avg_loss
    scores['loss_bce'] = avg_bce
    scores['loss_dice'] = avg_dice

    return scores


def run_test(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)

    # Load data YAML
    data_yaml = check_file(opt.data)
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    # Load hyp YAML if provided
    hyp_dict = {}
    if opt.hyp and Path(opt.hyp).exists():
        with open(opt.hyp, 'r') as f:
            hyp_dict = yaml.safe_load(f)

    # Resolve inputs, in_channels, and model architecture
    fusion = getattr(opt, 'fusion', 'concat')
    loss_cfg = {}

    if opt.weights and Path(opt.weights).exists():
        ckpt = torch_load(opt.weights, map_location=device)
        model_cfg = ckpt.get('cfg', opt.cfg)
        fusion = ckpt.get('fusion', fusion)
        inputs = ckpt.get('inputs', resolve_inputs(opt.inputs, data_dict.get('presets', {})))
        in_channels = ckpt.get('in_channels', get_channel_count(inputs, data_dict.get('presets', {}), fusion=fusion))
        nc = ckpt.get('nc', data_dict.get('nc', 1))
        model = Model(cfg=model_cfg, ch=in_channels, nc=nc).to(device)
        model.load_state_dict(ckpt['model'])
        ep_info = f" (Epoch {ckpt['epoch']})" if isinstance(ckpt, dict) and 'epoch' in ckpt else ""
        logger.info(f"Loaded weights from {opt.weights}{ep_info}")

        # Extract checkpoint loss configuration if available
        loss_cfg = ckpt.get('loss_cfg', {})
        if not loss_cfg and 'hyp' in ckpt:
            h = ckpt['hyp']
            loss_cfg = {
                'alpha': h.get('bce_weight', 1.0),
                'beta': h.get('dice_weight', 1.0),
                'pos_weight': h.get('pos_weight', 1.0)
            }
    else:
        inputs = resolve_inputs(opt.inputs, data_dict.get('presets', {}))
        in_channels = get_channel_count(inputs, data_dict.get('presets', {}), fusion=fusion)
        nc = data_dict.get('nc', 1)
        logger.info(f"Initializing model from config: {opt.cfg} (no pretrained weights)")
        model = Model(cfg=opt.cfg, ch=in_channels, nc=nc).to(device)

    # Determine loss parameters (Checkpoint loss_cfg -> CLI/hyp -> Defaults)
    alpha = float(hyp_dict.get('bce_weight', loss_cfg.get('alpha', 0.5 if opt.hyp else loss_cfg.get('alpha', 1.0))))
    beta = float(hyp_dict.get('dice_weight', loss_cfg.get('beta', 1.0)))
    pos_weight = float(hyp_dict.get('pos_weight', loss_cfg.get('pos_weight', 4.0 if opt.hyp else loss_cfg.get('pos_weight', 1.0))))
    smooth = float(loss_cfg.get('smooth', 1.0))

    # Criterion
    criterion = BCEDiceLoss(alpha=alpha, beta=beta, pos_weight=pos_weight, smooth=smooth).to(device)
    logger.info(f"Criterion configured: Total Loss = {alpha} * CrossEntropy (pos_weight={pos_weight}) + {beta} * Dice (smooth={smooth})")

    # DataLoader
    dataloader, dataset = create_dataloader(
        data_yaml_path=data_dict,
        split=opt.split,
        inputs=inputs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        augment=False,
        shuffle=False,
        num_workers=opt.workers,
        fusion=fusion
    )

    logger.info(f"Evaluating {len(dataset)} samples across modalities: {inputs} (channels={in_channels}, fusion={fusion})")

    # Run evaluation
    scores = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        conf_thres=opt.conf_thres,
        save_dir=save_dir,
        plots=opt.plots,
        pbar_desc=f"Testing [{opt.split}]"
    )

    # Print summary table
    print("\n" + "=" * 80)
    print(colorstr('bold', f"EVALUATION RESULTS [{opt.split.upper()}] (Threshold = {opt.conf_thres})"))
    print("=" * 80)
    print(f"  Loss Formulation : Total Loss = {alpha} * CrossEntropy (pos_weight={pos_weight}) + {beta} * Dice")
    print(f"  Total Loss       : {scores['loss']:.4f} (CE: {scores['loss_bce']:.4f}, Dice Loss: {scores['loss_dice']:.4f})")
    print("-" * 80)
    print(colorstr('bold', 'cyan', "  [Intersection-over-Union (IoU)]"))
    print(f"    • Foreground IoU (Landslide)  : {scores['fg_iou'] * 100:6.2f}%  (Micro/Global TP/(TP+FP+FN))")
    print(f"    • Per-Image Fg IoU (Macro)    : {scores['fg_iou_macro'] * 100:6.2f}%  (Mean of per-tile IoU scores)")
    print(f"    • Background IoU              : {scores['bg_iou'] * 100:6.2f}%  (TN/(TN+FP+FN))")
    print(colorstr('bold', 'bright_yellow', f"    • Two-Class Mean IoU (mIoU)   : {scores['miou'] * 100:6.2f}%  (Average of Fg & Bg IoU)"))
    print("-" * 80)
    print(colorstr('bold', 'cyan', "  [Dice Coefficient / F1-Score]"))
    print(f"    • Foreground Dice (Micro F1)  : {scores['fg_dice'] * 100:6.2f}%  (Global aggregate across dataset)")
    print(f"    • Foreground Dice (Macro)     : {scores['fg_dice_macro'] * 100:6.2f}%  (Per-tile mean, equal tile weight)")
    print(f"    • Background Dice             : {scores['bg_dice'] * 100:6.2f}%")
    print(colorstr('bold', 'bright_yellow', f"    • Two-Class Mean Dice (mDice) : {scores['mdice'] * 100:6.2f}%  (Average of Fg & Bg Dice)"))
    print("-" * 80)
    print(colorstr('bold', 'cyan', "  [Classification & Pixel Metrics]"))
    print(f"    • Foreground Precision        : {scores['precision'] * 100:6.2f}%")
    print(f"    • Foreground Recall           : {scores['recall'] * 100:6.2f}%")
    print(f"    • Overall Pixel Accuracy      : {scores['accuracy'] * 100:6.2f}%")
    print("=" * 80 + "\n")

    return scores


def parse_opt():
    parser = argparse.ArgumentParser(description="Evaluate Landslide Segmentation Model")
    parser.add_argument('--config', type=str, default='', help='path to yaml configuration file (e.g. configs/test.yaml)')
    parser.add_argument('--weights', type=str, default='', help='model weights path (.pt)')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters yaml path (e.g. data/hyp.scratch.yaml)')
    parser.add_argument('--inputs', type=str, default='rgb_only',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all) or list')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add'],
                        help='modality fusion mode: concat (channel concatenation) or add (element-wise addition into RGB)')
    parser.add_argument('--split', type=str, default='val', help='dataset split to evaluate (val, train)')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=512, help='inference image size')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='binary classification threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--project', default='runs/val', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--plots', action='store_true', default=True, help='save prediction visual plots')
    return parse_options_with_config(parser)


if __name__ == '__main__':
    opt = parse_opt()
    run_test(opt)
