import os
import sys
import time
import argparse
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.model_parser import Model
from data.datasets import create_dataloader, resolve_inputs, get_channel_count
from utils.loss import BCEDiceLoss
from utils.metrics import SegmentationMetrics
from utils.plots import plot_results, plot_predictions
from utils.torch_utils import select_device, init_seeds, model_info, EarlyStopping
from utils.general import increment_path, colorstr, check_file, set_logging
from test import evaluate

logger = set_logging(__name__)


def train(hyp, opt, device):
    t0 = time.time()

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_pt = weights_dir / 'last.pt'
    best_pt = weights_dir / 'best.pt'
    results_csv = save_dir / 'results.csv'

    # Save run arguments and hyperparameters
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)

    # Initialize seeds
    init_seeds(opt.seed)

    # Load dataset configuration
    data_yaml = check_file(opt.data)
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    # Resolve active input modalities and total input channels
    presets = data_dict.get('presets', {})
    inputs = resolve_inputs(opt.inputs, presets)
    in_channels = get_channel_count(inputs, presets)
    nc = data_dict.get('nc', 1)

    print("\n" + "=" * 75)
    print(colorstr('bold', 'cyan', '[START] STARTING LANDSLIDE SEGMENTATION TRAINING'))
    print("=" * 75)
    print(f"  Configuration File : {opt.cfg}")
    print(f"  Input Modalities   : {inputs} (Total Channels: {in_channels})")
    print(f"  Target Classes     : {nc} ({data_dict.get('names', ['landslide'])})")
    print(f"  Image Resolution   : {opt.img_size}x{opt.img_size}")
    print(f"  Batch Size         : {opt.batch_size}")
    print(f"  Total Epochs       : {opt.epochs}")
    print(f"  Save Directory     : {save_dir}")
    print("-" * 75)

    # Step 1: Dataloaders
    print(colorstr('bold', '[1/5] Loading datasets...'))
    train_loader, train_dataset = create_dataloader(
        data_yaml_path=data_dict,
        split='train',
        inputs=inputs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        augment=True,
        hyp=hyp,
        shuffle=True,
        num_workers=opt.workers
    )

    val_loader, val_dataset = create_dataloader(
        data_yaml_path=data_dict,
        split='val',
        inputs=inputs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        augment=False,
        shuffle=False,
        num_workers=opt.workers
    )
    print(f"      -> Train Samples: {len(train_dataset)} ({len(train_loader)} batches/epoch)")
    print(f"      -> Val Samples  : {len(val_dataset)} ({len(val_loader)} batches/epoch)")

    # Step 2: Build Model
    print(colorstr('bold', '[2/5] Building model architecture...'))
    model = Model(cfg=opt.cfg, ch=in_channels, nc=nc).to(device)

    # Load pretrained weights if provided
    if opt.weights and Path(opt.weights).exists():
        print(f"      -> Loading pretrained weights from {opt.weights}...")
        ckpt = torch.load(opt.weights, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"      -> Successfully transferred {len(pretrained_dict)}/{len(model_dict)} layers")

    # Step 3: Criterion
    print(colorstr('bold', '[3/5] Setting up loss function and optimizer...'))
    bce_w = hyp.get('bce_weight', 1.0)
    dice_w = hyp.get('dice_weight', 1.0)
    pos_w = hyp.get('pos_weight', 1.0)
    criterion = BCEDiceLoss(alpha=bce_w, beta=dice_w, pos_weight=pos_w).to(device)
    print(f"      -> Loss: {bce_w}*BCE + {dice_w}*Dice")

    # Optimizer & LR Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hyp.get('lr0', 1e-3),
        weight_decay=hyp.get('weight_decay', 1e-4)
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs,
        eta_min=hyp.get('lr0', 1e-3) * hyp.get('lrf', 0.01)
    )
    print(f"      -> Optimizer: AdamW (lr0={hyp.get('lr0', 1e-3)}, weight_decay={hyp.get('weight_decay', 1e-4)})")

    # Step 4: Mixed Precision
    print(colorstr('bold', '[4/5] Initializing compute hardware...'))
    cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=cuda) if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler(enabled=cuda)
    print(f"      -> Device: {device} | CUDA Accelerated: {cuda} | AMP Mixed Precision: {cuda}")

    # Early stopping helper
    early_stopping = EarlyStopping(patience=opt.patience, verbose=False, mode='max')

    # Logging headers
    with open(results_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss,val_iou,val_dice,val_precision,val_recall,val_accuracy\n')

    best_dice = 0.0
    best_iou = 0.0

    # Step 5: Training Loop
    print("\n" + colorstr('bold', '[5/5] Commencing training loop...\n'))
    header_str = f"{'Epoch':>7} | {'Train Loss':>10} | {'Val Loss':>10} | {'mIoU':>8} | {'Dice (F1)':>9} | {'Prec':>8} | {'Recall':>8} | {'ETA':>8}"
    print(header_str)
    print("-" * len(header_str))

    for epoch in range(1, opt.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_dice = 0.0
        optimizer.zero_grad()

        cur_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(
            train_loader,
            desc=f"[{epoch:03d}/{opt.epochs:03d}]",
            total=len(train_loader),
            bar_format="{l_bar}{bar:25}{r_bar}"
        )

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

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            b_loss = loss_dict['loss'].item()
            b_bce = loss_dict['bce'].item()
            b_dice = loss_dict['dice'].item()

            train_loss += b_loss
            train_bce += b_bce
            train_dice += b_dice

            mem = f"{torch.cuda.memory_reserved() / 1E9:.2f}G" if cuda else "CPU"
            pbar.set_postfix({
                'loss': f"{b_loss:.4f}",
                'dice': f"{1.0 - b_dice:.4f}",
                'mem': mem,
                'lr': f"{cur_lr:.1e}"
            })

        scheduler.step()
        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation phase
        val_scores = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            conf_thres=opt.conf_thres,
            save_dir=save_dir if epoch == opt.epochs else None,
            plots=(epoch == opt.epochs or epoch == 1),
            pbar_desc=f"Val [{epoch:03d}/{opt.epochs:03d}]"
        )

        val_loss = val_scores['loss']
        val_iou = val_scores['iou']
        val_dice = val_scores['dice']
        val_prec = val_scores['precision']
        val_rec = val_scores['recall']
        val_acc = val_scores['accuracy']

        # Log results to CSV
        with open(results_csv, 'a') as f:
            f.write(f"{epoch},{avg_train_loss:.5f},{val_loss:.5f},{val_iou:.5f},{val_dice:.5f},{val_prec:.5f},{val_rec:.5f},{val_acc:.5f}\n")

        # Calculate epoch duration and ETA
        epoch_time = time.time() - epoch_t0
        remaining_epochs = opt.epochs - epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_str = f"{int(eta_seconds // 60):02d}m{int(eta_seconds % 60):02d}s" if eta_seconds > 0 else "00m00s"

        # Formatted row output
        is_best = val_dice > best_dice
        row_str = (
            f"{epoch:3d}/{opt.epochs:3d} | "
            f"{avg_train_loss:10.4f} | "
            f"{val_loss:10.4f} | "
            f"{val_iou*100:7.2f}% | "
            f"{val_dice*100:8.2f}% | "
            f"{val_prec*100:7.2f}% | "
            f"{val_rec*100:7.2f}% | "
            f"{eta_str:>8}"
        )
        if is_best:
            best_dice = val_dice
            best_iou = val_iou
            row_str += colorstr('bright_green', ' (* Best)')
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_iou': best_iou,
                'cfg': opt.cfg,
                'inputs': inputs,
                'in_channels': in_channels,
                'nc': nc
            }, best_pt)

        print(row_str)

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'cfg': opt.cfg,
            'inputs': inputs,
            'in_channels': in_channels,
            'nc': nc
        }, last_pt)

        # Early stopping check
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print(colorstr('yellow', f"\n[INFO] Early stopping triggered at epoch {epoch} (no improvement for {opt.patience} epochs)."))
            break

    # Training Completion Summary
    total_time = (time.time() - t0) / 60
    print("\n" + "=" * 75)
    print(colorstr('bold', 'green', '[DONE] TRAINING COMPLETE!'))
    print("=" * 75)
    print(f"  Total Training Time : {total_time:.2f} minutes")
    print(f"  Best Validation Dice: {best_dice * 100:.2f}%")
    print(f"  Best Validation mIoU: {best_iou * 100:.2f}%")
    print(f"  Saved Best Checkpoint: {best_pt}")
    print(f"  Saved Last Checkpoint: {last_pt}")
    print(f"  Saved Results Log   : {results_csv}")
    print("=" * 75 + "\n")

    # Plot metrics
    plot_results(results_csv, save_dir=save_dir)


def parse_opt():
    parser = argparse.ArgumentParser(description="Train Landslide Segmentation Model")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters yaml path')
    parser.add_argument('--inputs', type=str, default='rgb_only',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all) or explicit list')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=512, help='image resolution')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='validation binary threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    # Load hyperparameters
    hyp_yaml = check_file(opt.hyp)
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.safe_load(f)

    device = select_device(opt.device, batch_size=opt.batch_size)
    train(hyp, opt, device)
