import os
import sys
import glob
import time
import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
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
from utils.torch_utils import select_device, init_seeds, model_info, EarlyStopping, load_pretrained_weights, get_rng_states, set_rng_states, torch_load
from utils.general import increment_path, colorstr, check_file, set_logging, parse_options_with_config
from test import evaluate

logger = set_logging(__name__)


def get_latest_run_checkpoint(project='runs/train'):
    """Search for the most recently modified last.pt in the runs directory."""
    last_pts = sorted(glob.glob(f"{project}/**/weights/last.pt", recursive=True), key=os.path.getmtime)
    if last_pts:
        return last_pts[-1]
    return None


def train(hyp, opt, device):
    t0 = time.time()
    resume = bool(opt.resume)
    start_epoch = 1
    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    best_scores = {}

    # -------------------------------------------------------------------------
    # Handle Resume Setup
    # -------------------------------------------------------------------------
    if resume:
        resume_path = opt.resume if isinstance(opt.resume, str) and opt.resume != 'get_last' else get_latest_run_checkpoint(opt.project)
        if not resume_path or not Path(resume_path).exists():
            raise FileNotFoundError(f"Cannot resume training: checkpoint not found at '{resume_path}'")

        ckpt = torch_load(resume_path, map_location=device)
        save_dir = Path(resume_path).parent.parent
        weights_dir = save_dir / 'weights'
        last_pt = weights_dir / 'last.pt'
        best_pt = weights_dir / 'best.pt'
        results_csv = save_dir / 'results.csv'

        # Load original options and hyperparameters if available
        opt_yaml = save_dir / 'opt.yaml'
        if opt_yaml.exists():
            with open(opt_yaml, 'r') as f:
                saved_opt = yaml.safe_load(f)
                opt.cfg = saved_opt.get('cfg', opt.cfg)
                opt.data = saved_opt.get('data', opt.data)
                opt.inputs = saved_opt.get('inputs', opt.inputs)
                opt.fusion = saved_opt.get('fusion', getattr(opt, 'fusion', 'concat'))
                opt.img_size = saved_opt.get('img_size', opt.img_size)

        start_epoch = ckpt.get('epoch', 0) + 1
        best_dice = ckpt.get('best_dice', ckpt.get('val_dice', 0.0))
        best_iou = ckpt.get('best_iou', ckpt.get('val_iou', 0.0))
        best_epoch = ckpt.get('best_epoch', ckpt.get('epoch', 0))
        best_scores = ckpt.get('best_scores', {})

        if 'rng_state' in ckpt and ckpt['rng_state'] is not None:
            set_rng_states(ckpt['rng_state'])

        print(colorstr('bold', 'yellow', f"\n[RESUME] Resuming training from {resume_path}"))
        print(f"  Resuming Epoch     : {start_epoch}/{opt.epochs}")
        print(f"  Previous Best Dice : {best_dice * 100:.2f}%, Best mIoU: {best_iou * 100:.2f}% (Epoch {best_epoch})")
        print(f"  Run Directory      : {save_dir}\n")

    else:
        # New training run directories
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

    # Initialize seeds (enables cuDNN benchmark by default for accelerated convolutions)
    init_seeds(opt.seed, deterministic=opt.deterministic)

    # Load dataset configuration
    data_yaml = check_file(opt.data)
    with open(data_yaml, 'r') as f:
        data_dict = yaml.safe_load(f)

    # Resolve active input modalities and total input channels
    presets = data_dict.get('presets', {})
    inputs = resolve_inputs(opt.inputs, presets)
    in_channels = get_channel_count(inputs, presets, fusion=opt.fusion)
    nc = data_dict.get('nc', 1)

    if not resume:
        print("\n" + "=" * 75)
        print(colorstr('bold', 'cyan', '[START] STARTING LANDSLIDE SEGMENTATION TRAINING'))
        print("=" * 75)
        print(f"  Configuration File : {opt.cfg}")
        print(f"  Input Modalities   : {inputs} (Total Channels: {in_channels}, Fusion: {opt.fusion})")
        print(f"  Target Classes     : {nc} ({data_dict.get('names', ['landslide'])})")
        print(f"  Image Resolution   : {opt.img_size}x{opt.img_size}")
        print(f"  Batch Size         : {opt.batch_size}")
        print(f"  RAM Caching        : {'Enabled (Zero Disk I/O)' if opt.cache_ram else 'Disabled'}")
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
        num_workers=opt.workers,
        cache_ram=opt.cache_ram,
        fusion=opt.fusion
    )

    val_loader, val_dataset = create_dataloader(
        data_yaml_path=data_dict,
        split='val',
        inputs=inputs,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        augment=False,
        shuffle=False,
        num_workers=opt.workers,
        cache_ram=opt.cache_ram,
        fusion=opt.fusion
    )
    print(f"      -> Train Samples: {len(train_dataset)} ({len(train_loader)} batches/epoch)")
    print(f"      -> Val Samples  : {len(val_dataset)} ({len(val_loader)} batches/epoch)")


    # Step 2: Build Model
    print(colorstr('bold', '[2/5] Building model architecture...'))
    model = Model(cfg=opt.cfg, ch=in_channels, nc=nc).to(device)

    # Load weights (either from resume checkpoint or pretrained weights)
    if resume:
        model.load_state_dict(ckpt['model'])
        print(f"      -> Restored full model weights from checkpoint")
    elif opt.weights:
        print(f"      -> Loading pretrained weights from {opt.weights}...")
        res = load_pretrained_weights(model, opt.weights, device=device)
        print(f"      -> Transferred {res['matched']}/{res['total']} matching layers from {res['source']} for fine-tuning")

    # Step 3: Criterion
    print(colorstr('bold', '[3/5] Setting up loss function and optimizer...'))
    bce_w = hyp.get('bce_weight', 1.0)
    dice_w = hyp.get('dice_weight', 1.0)
    pos_w = hyp.get('pos_weight', 1.0)
    criterion = BCEDiceLoss(alpha=bce_w, beta=dice_w, pos_weight=pos_w).to(device)
    print(f"      -> Loss: {bce_w}*BCE + {dice_w}*Dice")

    # Optimizer with parameter groups & momentum
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases (no decay)
        if isinstance(v, (nn.BatchNorm2d, nn.GroupNorm)):
            pg1.append(v.weight)  # batchnorm weights (no decay)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg0.append(v.weight)  # weights (apply weight decay)

    optimizer = optim.AdamW(
        pg0,
        lr=hyp.get('lr0', 1e-3),
        betas=(hyp.get('momentum', 0.937), 0.999),
        weight_decay=hyp.get('weight_decay', 1e-4)
    )
    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': pg2, 'weight_decay': 0.0})

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=opt.epochs,
        eta_min=hyp.get('lr0', 1e-3) * hyp.get('lrf', 0.01)
    )

    if resume:
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            print(f"      -> Restored optimizer state and learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
            print("      -> Restored LR scheduler state (CosineAnnealingLR)")
        else:
            for _ in range(start_epoch - 1):
                scheduler.step()
    else:
        print(f"      -> Optimizer: AdamW (lr0={hyp.get('lr0', 1e-3)}, momentum={hyp.get('momentum', 0.937)}, weight_decay={hyp.get('weight_decay', 1e-4)})")

    # Step 4: Mixed Precision
    print(colorstr('bold', '[4/5] Initializing compute hardware...'))
    cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=cuda) if hasattr(torch, 'amp') else torch.cuda.amp.GradScaler(enabled=cuda)
    if resume and 'scaler' in ckpt and ckpt['scaler'] is not None and cuda:
        try:
            scaler.load_state_dict(ckpt['scaler'])
            print("      -> Restored AMP GradScaler dynamic scaling state")
        except Exception as e:
            logger.warning(f"Could not restore GradScaler state: {e}")
    print(f"      -> Device: {device} | CUDA Accelerated: {cuda} | AMP Mixed Precision: {cuda}")

    # Early stopping helper
    early_stopping = EarlyStopping(patience=opt.patience, verbose=False, mode='max')
    if resume and 'early_stopping' in ckpt and ckpt['early_stopping'] is not None:
        early_stopping.load_state_dict(ckpt['early_stopping'])
        print(f"      -> Restored EarlyStopping state: counter={early_stopping.counter}/{early_stopping.patience}, best_score={early_stopping.best_score}")

    # Logging headers (create new or continue appending if resuming)
    if not resume or not results_csv.exists():
        with open(results_csv, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_fg_iou,val_miou,val_fg_dice,val_fg_dice_macro,val_precision,val_recall,val_accuracy\n')

    # Warmup setup
    warmup_epochs = hyp.get('warmup_epochs', 0.0)
    nw = max(round(warmup_epochs * len(train_loader)), 10) if warmup_epochs > 0 else 0
    w_mom0 = hyp.get('warmup_momentum', 0.8)
    mom = hyp.get('momentum', 0.937)
    w_bias_lr = hyp.get('warmup_bias_lr', 0.1)
    base_lr = hyp.get('lr0', 1e-3)

    # Step 5: Training Loop
    print("\n" + colorstr('bold', '[5/5] Commencing training loop...\n'))
    header_str = f"{'Epoch':>7} | {'Train Loss':>10} | {'Val Loss':>10} | {'Fg IoU':>8} | {'mIoU':>8} | {'Fg Dice':>8} | {'Prec':>8} | {'Recall':>8} | {'ETA':>8}"
    print(header_str)
    print("-" * len(header_str))

    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_dice = 0.0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"[{epoch:03d}/{opt.epochs:03d}]",
            total=len(train_loader),
            bar_format="{l_bar}{bar:25}{r_bar}"
        )

        accumulate = max(1, opt.accumulate)
        for batch_idx, (tensors, targets, stems) in enumerate(pbar):
            ni = (epoch - 1) * len(train_loader) + batch_idx  # integrated batch index

            # Linear warmup for LR and momentum
            if ni <= nw and nw > 0:
                xi = [0, nw]
                for j, x in enumerate(optimizer.param_groups):
                    if j == 2:  # bias group
                        x['lr'] = float(np.interp(ni, xi, [w_bias_lr, base_lr]))
                    else:
                        x['lr'] = float(np.interp(ni, xi, [0.0, base_lr]))
                    if 'betas' in x:
                        x['betas'] = (float(np.interp(ni, xi, [w_mom0, mom])), 0.999)

            cur_lr = optimizer.param_groups[0]['lr']
            tensors = tensors.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Determine actual sub-batch group size for unbiased normalization on trailing batches
            remaining_in_epoch = len(train_loader) - (batch_idx // accumulate) * accumulate
            group_size = min(accumulate, remaining_in_epoch)

            if cuda:
                with torch.amp.autocast('cuda'):
                    logits = model(tensors)
                    loss, loss_dict = criterion(logits, targets)
                    if group_size > 1:
                        loss = loss / group_size
            else:
                logits = model(tensors)
                loss, loss_dict = criterion(logits, targets)
                if group_size > 1:
                    loss = loss / group_size

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulate == 0 or (batch_idx + 1) == len(train_loader):
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
        val_fg_iou = val_scores.get('fg_iou', val_scores.get('iou', 0.0))
        val_miou = val_scores.get('miou', val_fg_iou)
        val_fg_dice = val_scores.get('fg_dice', val_scores.get('dice', 0.0))
        val_fg_dice_macro = val_scores.get('fg_dice_macro', val_fg_dice)
        val_prec = val_scores['precision']
        val_rec = val_scores['recall']
        val_acc = val_scores['accuracy']

        # Log results to CSV
        with open(results_csv, 'a') as f:
            f.write(f"{epoch},{avg_train_loss:.5f},{val_loss:.5f},{val_fg_iou:.5f},{val_miou:.5f},{val_fg_dice:.5f},{val_fg_dice_macro:.5f},{val_prec:.5f},{val_rec:.5f},{val_acc:.5f}\n")

        # Calculate epoch duration and ETA
        epoch_time = time.time() - epoch_t0
        remaining_epochs = opt.epochs - epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_str = f"{int(eta_seconds // 60):02d}m{int(eta_seconds % 60):02d}s" if eta_seconds > 0 else "00m00s"

        # Checkpoint payload
        ckpt_payload = {
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_scores': best_scores,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict() if cuda and hasattr(scaler, 'state_dict') else None,
            'early_stopping': early_stopping.state_dict(),
            'rng_state': get_rng_states(),
            'val_dice': val_fg_dice,
            'val_iou': val_fg_iou,
            'val_miou': val_miou,
            'best_dice': best_dice,
            'best_iou': best_iou,
            'cfg': opt.cfg,
            'inputs': inputs,
            'fusion': opt.fusion,
            'in_channels': in_channels,
            'nc': nc,
            'hyp': hyp,
            'loss_cfg': {
                'alpha': float(bce_w),
                'beta': float(dice_w),
                'pos_weight': float(pos_w),
                'smooth': 1.0
            }
        }

        # Formatted row output
        is_best = val_fg_dice > best_dice
        row_str = (
            f"{epoch:3d}/{opt.epochs:3d} | "
            f"{avg_train_loss:10.4f} | "
            f"{val_loss:10.4f} | "
            f"{val_fg_iou*100:7.2f}% | "
            f"{val_miou*100:7.2f}% | "
            f"{val_fg_dice*100:7.2f}% | "
            f"{val_prec*100:7.2f}% | "
            f"{val_rec*100:7.2f}% | "
            f"{eta_str:>8}"
        )
        if is_best:
            best_dice = val_fg_dice
            best_iou = val_fg_iou
            best_epoch = epoch
            best_scores = {
                'epoch': epoch,
                'train_loss': float(avg_train_loss),
                'val_loss': float(val_loss),
                'fg_iou': float(val_fg_iou),
                'bg_iou': float(val_scores.get('bg_iou', 0.0)),
                'miou': float(val_miou),
                'fg_dice': float(val_fg_dice),
                'fg_dice_macro': float(val_fg_dice_macro),
                'bg_dice': float(val_scores.get('bg_dice', 0.0)),
                'mdice': float(val_scores.get('mdice', 0.0)),
                'precision': float(val_prec),
                'recall': float(val_rec),
                'accuracy': float(val_acc)
            }
            ckpt_payload['best_epoch'] = best_epoch
            ckpt_payload['best_scores'] = best_scores
            ckpt_payload['best_dice'] = best_dice
            ckpt_payload['best_iou'] = best_iou
            row_str += colorstr('bright_green', ' (* Best)')
            torch.save(ckpt_payload, best_pt)

        print(row_str)

        # Save latest checkpoint
        torch.save(ckpt_payload, last_pt)

        # Early stopping check
        early_stopping(val_fg_dice)
        if early_stopping.early_stop:
            print(colorstr('yellow', f"\n[INFO] Early stopping triggered at epoch {epoch} (no improvement for {opt.patience} epochs)."))
            break

    # Training Completion Summary
    total_time = (time.time() - t0) / 60
    total_epochs_trained = epoch - start_epoch + 1
    print("\n" + "=" * 78)
    print(colorstr('bold', 'bright_green', '[DONE] TRAINING COMPLETE!'))
    print("=" * 78)
    print(f"  Total Training Time : {total_time:.2f} minutes ({total_time / max(total_epochs_trained, 1):.2f} min/epoch)")
    print(f"  Total Epochs Trained: {total_epochs_trained}")
    print("-" * 78)
    print(colorstr('bold', 'bright_yellow', f"  ★ BEST VALIDATION METRICS (Achieved at Epoch {best_epoch}):"))
    print(colorstr('bold', 'bright_green',  f"    • Foreground IoU (Landslide)  : {best_iou * 100:6.2f}% (Global Micro)"))
    print(colorstr('bold', 'bright_green',  f"    • Two-Class Mean IoU (mIoU)   : {best_scores.get('miou', 0.0) * 100:6.2f}%"))
    print(colorstr('bold', 'bright_green',  f"    • Foreground Dice (Micro F1)  : {best_dice * 100:6.2f}% (Global Micro)"))
    print(colorstr('bold', 'bright_green',  f"    • Foreground Dice (Macro)     : {best_scores.get('fg_dice_macro', 0.0) * 100:6.2f}% (Per-Image Mean)"))
    print(colorstr('bold', 'bright_green',  f"    • Two-Class Mean Dice (mDice) : {best_scores.get('mdice', 0.0) * 100:6.2f}%"))
    if best_scores:
        print(f"    • Precision @ Best Epoch      : {best_scores.get('precision', 0.0) * 100:6.2f}%")
        print(f"    • Recall @ Best Epoch         : {best_scores.get('recall', 0.0) * 100:6.2f}%")
        print(f"    • Pixel Accuracy @ Best Epoch : {best_scores.get('accuracy', 0.0) * 100:6.2f}%")
        print(f"    • Val Loss @ Best Epoch       : {best_scores.get('val_loss', 0.0):.4f}")
        print(f"    • Train Loss @ Best Epoch     : {best_scores.get('train_loss', 0.0):.4f}")
    print("-" * 78)
    print(f"  Artifacts & Checkpoints:")
    print(f"    • Best Checkpoint   : {best_pt}")
    print(f"    • Last Checkpoint   : {last_pt}")
    print(f"    • Results CSV Log   : {results_csv}")
    print(f"    • Best Summary Log  : {save_dir / 'best_metrics.yaml'}")
    print(f"    • Visual Curves     : {save_dir / 'results.png'}")
    print("=" * 78 + "\n")

    # Save best metrics summary to YAML
    if best_scores:
        best_summary_file = save_dir / 'best_metrics.yaml'
        with open(best_summary_file, 'w') as f:
            yaml.safe_dump({
                'best_epoch': int(best_epoch),
                'best_fg_iou': float(best_iou),
                'best_miou': float(best_scores.get('miou', 0.0)),
                'best_bg_iou': float(best_scores.get('bg_iou', 0.0)),
                'best_fg_dice': float(best_dice),
                'best_fg_dice_macro': float(best_scores.get('fg_dice_macro', 0.0)),
                'best_mdice': float(best_scores.get('mdice', 0.0)),
                'best_precision': float(best_scores.get('precision', 0.0)),
                'best_recall': float(best_scores.get('recall', 0.0)),
                'best_accuracy': float(best_scores.get('accuracy', 0.0)),
                'val_loss': float(best_scores.get('val_loss', 0.0)),
                'train_loss': float(best_scores.get('train_loss', 0.0)),
                'total_epochs': int(epoch),
                'training_time_minutes': round(float(total_time), 2)
            }, f, sort_keys=False)

    # Plot metrics
    plot_results(results_csv, save_dir=save_dir)

    return best_dice, best_iou, best_scores


def parse_opt():
    parser = argparse.ArgumentParser(description="Train Landslide Segmentation Model")
    parser.add_argument('--config', type=str, default='', help='path to yaml configuration file (e.g. configs/train.yaml)')
    parser.add_argument('--weights', type=str, default='', help='initial pretrained weights path (e.g. weights.pt or unet_carvana)')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume most recent training run or specify path to last.pt (e.g. --resume or --resume runs/train/exp/weights/last.pt)')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model.yaml architecture path')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters yaml path')
    parser.add_argument('--inputs', type=str, default='rgb_only',
                        help='input option preset (rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all, rgb_add_dtm) or explicit list')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add'],
                        help='modality fusion mode: concat (channel concatenation) or add (element-wise addition into RGB)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--img-size', type=int, default=512, help='image resolution')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='validation binary threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=2, help='dataloader workers (2-4 recommended for Windows/CUDA)')
    parser.add_argument('--cache-ram', action='store_true', help='cache preprocessed dataset arrays in RAM for maximum GPU saturation')
    parser.add_argument('--deterministic', action='store_true', help='enable deterministic cuDNN (disables benchmark auto-tuner)')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--project', default='runs/train', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parse_options_with_config(parser)




if __name__ == '__main__':
    opt = parse_opt()

    # Load hyperparameters
    hyp_yaml = check_file(opt.hyp)
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.safe_load(f)

    device = select_device(opt.device, batch_size=opt.batch_size)
    train(hyp, opt, device)
