import os
import sys
import gc
import copy
import time
import argparse
import logging
import shutil
from pathlib import Path
import csv
import yaml
import numpy as np
import torch

from utils.general import increment_path, colorstr, check_file, set_logging, parse_options_with_config
from utils.torch_utils import select_device
from utils.plots import plot_evolve
from train import train, parse_opt as train_parse_opt

logger = set_logging(__name__)

# Search space definition with bounds, scales, and Gaussian perturbation sigmas
GENE_SPACE = {
    'lr0': {'min': 1e-4, 'max': 1e-2, 'log': True, 'sigma': 0.3},
    'lrf': {'min': 0.01, 'max': 0.5, 'log': True, 'sigma': 0.3},
    'momentum': {'min': 0.70, 'max': 0.98, 'log': False, 'sigma': 0.05},
    'weight_decay': {'min': 1e-6, 'max': 1e-2, 'log': True, 'sigma': 0.5},
    'warmup_epochs': {'min': 0.0, 'max': 5.0, 'log': False, 'sigma': 0.5},
    'bce_weight': {'min': 0.1, 'max': 2.0, 'log': False, 'sigma': 0.2},
    'dice_weight': {'min': 0.5, 'max': 3.0, 'log': False, 'sigma': 0.2},
    'pos_weight': {'min': 1.0, 'max': 15.0, 'log': False, 'sigma': 1.0},
    'fliplr': {'min': 0.0, 'max': 1.0, 'log': False, 'sigma': 0.15},
    'flipud': {'min': 0.0, 'max': 1.0, 'log': False, 'sigma': 0.15},
    'rot90': {'min': 0.0, 'max': 1.0, 'log': False, 'sigma': 0.15},
}


def compute_fitness(scores: dict, recall_floor: float = 0.05) -> float:
    """
    Calculate composite scalar fitness to guide evolutionary selection.

    Fitness = 0.5 * Dice + 0.3 * mIoU + 0.2 * Recall
    A recall floor penalty is enforced to reject degenerate all-background solutions.
    """
    dice = scores.get('dice', 0.0)
    iou = scores.get('iou', 0.0)
    recall = scores.get('recall', 0.0)

    if recall < recall_floor:
        return 0.0

    fitness = 0.5 * dice + 0.3 * iou + 0.2 * recall
    return float(max(0.0, fitness))


def mutate_genes(parent_hyp: dict, gene_space: dict = GENE_SPACE, prob: float = 0.8) -> dict:
    """
    Apply bounded Gaussian mutations to parent hyperparameters.
    """
    child = copy.deepcopy(parent_hyp)
    for gene, meta in gene_space.items():
        if gene not in child:
            continue

        if np.random.rand() < prob:
            curr_val = float(child[gene])
            if meta['log']:
                log_val = np.log10(max(curr_val, 1e-12))
                perturbed_log = log_val + np.random.normal(0, meta['sigma'])
                new_val = 10 ** perturbed_log
            else:
                new_val = curr_val + np.random.normal(0, meta['sigma'])

            # Clamp within specified search boundaries
            new_val = float(np.clip(new_val, meta['min'], meta['max']))

            if gene == 'warmup_epochs':
                new_val = round(new_val, 1)
            elif meta['log']:
                new_val = float(f"{new_val:.6e}")
            else:
                new_val = round(new_val, 4)

            child[gene] = new_val

    return child


def select_parent(history: list) -> dict:
    """
    Select an elite parent from evaluation history weighted by fitness^2.
    """
    if not history:
        raise ValueError("Cannot select parent from empty history.")

    # Sort history by fitness descending
    sorted_history = sorted(history, key=lambda x: x['fitness'], reverse=True)
    top_k = sorted_history[:min(5, len(sorted_history))]

    fitnesses = np.array([x['fitness'] for x in top_k], dtype=np.float64)
    total_fit = np.sum(fitnesses ** 2)

    if total_fit <= 1e-8:
        # If all top candidates have 0 fitness, sample uniformly
        idx = np.random.choice(len(top_k))
    else:
        probs = (fitnesses ** 2) / total_fit
        idx = np.random.choice(len(top_k), p=probs)

    return top_k[idx]['hyp']


def run_evolution(opt):
    t0 = time.time()
    device = select_device(opt.device, batch_size=opt.batch_size)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)
    evolve_csv = save_dir / 'evolve.csv'
    hyp_evolved_yaml = save_dir / 'hyp_evolved.yaml'
    gen_runs_dir = save_dir / 'generations'
    gen_runs_dir.mkdir(parents=True, exist_ok=True)

    # Save evolution run configuration
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Load baseline hyperparameters
    hyp_file = check_file(opt.hyp)
    with open(hyp_file, 'r') as f:
        baseline_hyp = yaml.safe_load(f)

    # Gene names in canonical order
    gene_names = list(GENE_SPACE.keys())
    metric_headers = ['generation', 'fitness', 'val_dice', 'val_iou', 'val_recall', 'val_precision', 'val_loss']
    csv_headers = metric_headers + gene_names

    history = []
    best_fitness = -1.0
    best_gen = 0
    start_gen = 0

    # Handle resume
    if opt.resume and evolve_csv.exists():
        with open(evolve_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                g = int(row['generation'])
                fit = float(row['fitness'])
                h = {k: float(row[k]) for k in gene_names if k in row}
                entry = {
                    'generation': g,
                    'fitness': fit,
                    'scores': {
                        'dice': float(row['val_dice']),
                        'iou': float(row['val_iou']),
                        'recall': float(row['val_recall']),
                        'precision': float(row['val_precision']),
                        'val_loss': float(row['val_loss'])
                    },
                    'hyp': h
                }
                history.append(entry)
                if fit > best_fitness:
                    best_fitness = fit
                    best_gen = g

        start_gen = len(history)
        print(colorstr('bold', 'yellow', f"\n[RESUME] Resuming evolution from {evolve_csv}"))
        print(f"  Loaded Generations : {len(history)} evaluated")
        print(f"  Current Best Score : Fitness = {best_fitness:.4f} (Gen {best_gen})\n")
    else:
        # Initialize CSV header
        with open(evolve_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    print("\n" + "=" * 80)
    print(colorstr('bold', 'cyan', "[EVOLVE] STARTING GENETIC HYPERPARAMETER EVOLUTION"))
    print("=" * 80)
    print(f"  Target Generations : {opt.generations}")
    print(f"  Screening Epochs   : {opt.epochs} epochs / generation")
    print(f"  Batch Size         : {opt.batch_size}")
    print(f"  Inputs Modalities  : {opt.inputs} (Fusion: {opt.fusion})")
    print(f"  Baseline Hyp File  : {opt.hyp}")
    print(f"  Output Directory   : {save_dir}")
    print("-" * 80 + "\n")

    # Evolutionary loop
    for gen in range(start_gen, opt.generations):
        gen_t0 = time.time()
        print("\n" + colorstr('bold', 'cyan', f"[Generation {gen + 1:02d}/{opt.generations:02d}]"))

        if gen == 0 and not history:
            # Generation 0: Evaluate baseline seed
            current_hyp = copy.deepcopy(baseline_hyp)
            print("  -> Evaluating baseline seed hyperparameters...")
        else:
            # Select parent and mutate
            parent_hyp = select_parent(history)
            current_hyp = mutate_genes(parent_hyp, prob=opt.mutation_prob)
            print("  -> Mutated genes from elite parent:")
            for g_name in gene_names:
                diff = current_hyp[g_name] - parent_hyp.get(g_name, current_hyp[g_name])
                diff_str = f" ({diff:+.4e})" if abs(diff) > 1e-9 else ""
                print(f"     • {g_name:<14s}: {current_hyp[g_name]}{diff_str}")

        # Configure candidate training run
        candidate_opt = copy.deepcopy(opt)
        candidate_opt.name = f"gen_{gen + 1:03d}"
        candidate_opt.project = str(gen_runs_dir)
        candidate_opt.epochs = opt.epochs
        candidate_opt.patience = max(4, opt.epochs // 2)
        candidate_opt.resume = False
        candidate_opt.exist_ok = True

        # Run candidate training
        try:
            best_dice, best_iou, best_scores = train(current_hyp, candidate_opt, device)
            fitness = compute_fitness(best_scores)
        except Exception as e:
            print(colorstr('red', f"  [Error] Candidate generation {gen + 1} failed during training: {e}"))
            best_scores = {'dice': 0.0, 'iou': 0.0, 'recall': 0.0, 'precision': 0.0, 'val_loss': 99.0}
            fitness = 0.0

        gen_time = (time.time() - gen_t0) / 60
        is_new_best = fitness > best_fitness

        print(f"\n  Generation {gen + 1:02d} Summary ({gen_time:.1f}m):")
        print(f"    • Composite Fitness : {fitness:.4f}" + (colorstr('bold', 'bright_green', ' ★ NEW ALL-TIME BEST!') if is_new_best else ""))
        print(f"    • Val Dice (F1)     : {best_scores.get('dice', 0.0) * 100:6.2f}%")
        print(f"    • Val mIoU          : {best_scores.get('iou', 0.0) * 100:6.2f}%")
        print(f"    • Val Recall        : {best_scores.get('recall', 0.0) * 100:6.2f}%")
        print(f"    • Val Precision     : {best_scores.get('precision', 0.0) * 100:6.2f}%")

        if is_new_best:
            best_fitness = fitness
            best_gen = gen + 1
            # Save updated evolved YAML
            with open(hyp_evolved_yaml, 'w') as f:
                yaml.safe_dump(current_hyp, f, sort_keys=False)
            print(colorstr('bright_green', f"    -> Exported updated best genes to: {hyp_evolved_yaml}"))

        # Append to history
        history.append({
            'generation': gen + 1,
            'fitness': fitness,
            'scores': best_scores,
            'hyp': current_hyp
        })

        # Append to CSV
        csv_row = [
            gen + 1,
            f"{fitness:.5f}",
            f"{best_scores.get('dice', 0.0):.5f}",
            f"{best_scores.get('iou', 0.0):.5f}",
            f"{best_scores.get('recall', 0.0):.5f}",
            f"{best_scores.get('precision', 0.0):.5f}",
            f"{best_scores.get('val_loss', 0.0):.5f}",
        ] + [str(current_hyp.get(k, '')) for k in gene_names]

        with open(evolve_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        # Plot evolution progress
        plot_evolve(evolve_csv, save_dir=save_dir)

        # Cleanup GPU memory and candidate run weights to conserve disk
        torch.cuda.empty_cache()
        gc.collect()

        gen_dir = gen_runs_dir / f"gen_{gen + 1:03d}"
        if gen_dir.exists() and not is_new_best:
            # Keep logs, delete heavy weights for non-best candidates
            weights_dir = gen_dir / 'weights'
            if weights_dir.exists():
                shutil.rmtree(weights_dir, ignore_errors=True)

    # Evolution Final Report
    total_time_hours = (time.time() - t0) / 3600
    print("\n" + "=" * 80)
    print(colorstr('bold', 'bright_green', '[DONE] HYPERPARAMETER EVOLUTION COMPLETE!'))
    print("=" * 80)
    print(f"  Total Duration      : {total_time_hours:.2f} hours")
    print(f"  Generations Evaluated: {len(history)}")
    print("-" * 80)
    print(colorstr('bold', 'bright_yellow', f"  ★ BEST EVOLVED INDIVIDUAL (Discovered at Generation {best_gen}):"))
    print(f"    • Composite Fitness : {best_fitness:.4f}")
    if history:
        best_entry = max(history, key=lambda x: x['fitness'])
        b_scores = best_entry['scores']
        print(f"    • Best Val Dice (F1): {b_scores.get('dice', 0.0) * 100:.2f}%")
        print(f"    • Best Val mIoU     : {b_scores.get('iou', 0.0) * 100:.2f}%")
        print(f"    • Best Val Recall   : {b_scores.get('recall', 0.0) * 100:.2f}%")
        print(f"    • Best Val Precision: {b_scores.get('precision', 0.0) * 100:.2f}%")
    print("-" * 80)
    print(f"  Generated Artifacts:")
    print(f"    • Best Evolved Hyp  : {hyp_evolved_yaml}")
    print(f"    • Evolution History : {evolve_csv}")
    print(f"    • Fitness Progress  : {save_dir / 'evolve_fitness.png'}")
    print(f"    • Gene Correlations : {save_dir / 'evolve_scatter.png'}")
    print("-" * 80)
    print("  To train a full model with the evolved hyperparameters, run:")
    print(colorstr('bold', 'bright_green', f"    python train.py --config configs/train.yaml --hyp {hyp_evolved_yaml} --epochs 50\n"))


def parse_opt():
    parser = argparse.ArgumentParser(description="Hyperparameter Evolution for Landslide Segmentation")
    parser.add_argument('--config', type=str, default='', help='path to configuration yaml (e.g. configs/evolve.yaml or configs/train.yaml)')
    parser.add_argument('--generations', type=int, default=30, help='total evolution generations')
    parser.add_argument('--epochs', type=int, default=15, help='screening epochs per candidate')
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size')
    parser.add_argument('--mutation-prob', type=float, default=0.8, help='gene mutation probability')
    parser.add_argument('--cfg', type=str, default='models/architectures/unet.yaml', help='model architecture yaml')
    parser.add_argument('--data', type=str, default='data/landslide.yaml', help='dataset yaml')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='baseline seed hyperparameters yaml')
    parser.add_argument('--weights', type=str, default='unet_carvana', help='initial weights (e.g. unet_carvana)')
    parser.add_argument('--inputs', type=str, default='rgb_dtm', help='input modality preset')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'add'], help='modality fusion mode')
    parser.add_argument('--img-size', type=int, default=512, help='image resolution')
    parser.add_argument('--device', default='', help='cuda device (e.g. 0) or cpu')
    parser.add_argument('--workers', type=int, default=2, help='dataloader workers')
    parser.add_argument('--cache-ram', action='store_true', help='cache preprocessed arrays in RAM')
    parser.add_argument('--deterministic', action='store_true', help='deterministic cuDNN')
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--project', default='runs/evolve', help='save directory project')
    parser.add_argument('--name', default='exp', help='save directory experiment name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--resume', action='store_true', help='resume evolution from existing evolve.csv')
    return parse_options_with_config(parser)


if __name__ == '__main__':
    opt = parse_opt()
    run_evolution(opt)
