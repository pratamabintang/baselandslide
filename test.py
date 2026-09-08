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
from utils.torch_utils import select_device
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

    # Resolve inputs and in_channels
    inputs = resolve_inputs(opt.inputs, data_dict.get('presets', {}))
    in_channels = get_channel_count(inputs, data_dict.get('presets', {}))

    # Initialize model
    if opt.weights and Path(opt.weights).exists():
        ckpt = torch.load(opt.weights, map_location=device)
        model_cfg = ckpt.get('cfg', opt.cfg)
        model = Model(cfg=model_cfg, ch=in_channels, nc=data_dict.get('nc', 1)).to(device)
        model.load_state_dict(ckpt['model'])
        logger.info(f"Loaded weights from {opt.weights}")
    else:
        logger.info(f"Initializing model from config: {opt.cfg} (no pretrained weights)")
        model = Model(cfg=opt.cfg, ch=in_channels, nc=data_dict.get('nc', 1)).to(device)

    # Criterion
    criterion = BCEDiceLoss(alpha=1.0, beta=1.0).to(device)

    # DataLoader
    dataloader, dataset = create_dataloader(
        data_yaml_path=data_dict,
        split=opt.split,
        inputs=inputs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        augment=False,
        shuffle=False,
        num_workers=opt.workers
    )

    logger.info(f"Evaluating {len(dataset)} samples across modalities: {inputs} (channels={in_channels})")

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
    print("\n" + "=" * 70)
    print(colorstr('bold', f"EVALUATION RESULTS [{opt.split.upper()}] (Threshold = {opt.conf_thres})"))
    print("=" * 70)
    print(f"  Total Loss     : {scores['loss']:.4f} (BCE: {scores['loss_bce']:.4f}, Dice: {scores['loss_dice']:.4f})")
    print(f"  Mean IoU       : {scores['iou'] * 100:.2f}%")
    print(f"  Dice (F1-Score): {scores['dice'] * 100:.2f}%")
    print(f"  Precision      : {scores['precision'] * 100:.2f}%")
    print(f"  Recall         : {scores['recall'] * 100:.2f}%")
    print(f"  Pixel Accuracy : {scores['accuracy'] * 100:.2f}%")
    print("=" * 70 + "\n")

    return scores


def parse_opt():
    parser = argparse.ArgumentParser(description="Evaluate Landslide Segmentation Model")
    parser.add_argument('--config', type=str, default='', help='path to yaml configuration file (e.g. configs/test.yaml)')
    parser.add_argument('--weights', type=str, default='', help='model weights path (.pt)')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--inputs', type=str, default='rgb_only',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all) or list')
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
