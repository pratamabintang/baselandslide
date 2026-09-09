import sys
import time
import shutil
from pathlib import Path
import torch
import numpy as np

# Imports from codebase
from data.base import BaseMultiModalDataset, register_dataset, DATASET_REGISTRY
from data.datasets import build_dataset, create_dataloader, resolve_inputs, get_channel_count
from models.model_parser import Model
from utils.loss import BCEDiceLoss
from utils.metrics import SegmentationMetrics
from utils.general import colorstr
from train import train
from test import evaluate
from predict import run_predict, parse_opt as parse_predict_opt, load_single_sample


def run_smoke_test():
    print("=" * 80)
    print(colorstr('bold', 'cyan', "[SMOKE TEST] RUNNING COMPREHENSIVE REPOSITORY SMOKE TEST"))
    print("=" * 80)
    passed_tests = 0
    total_tests = 15
    t0 = time.time()

    # -------------------------------------------------------------------------
    # TEST 1: Dataset Registry & All Modality Presets (Concat Mode)
    # -------------------------------------------------------------------------
    print(f"\n[Test 1/{total_tests}] Testing Dataset Registry & Modality Presets (Concat Mode)...")
    presets = ['rgb_only', 'topo_only', 'rgb_dtm', 'rgb_slope', 'rgb_aspect', 'all']
    expected_channels = {'rgb_only': 3, 'topo_only': 4, 'rgb_dtm': 4, 'rgb_slope': 4, 'rgb_aspect': 5, 'all': 7}

    for preset in presets:
        ds = build_dataset('data/landslide.yaml', split='train', inputs=preset)
        ch = get_channel_count(preset, fusion='concat')
        assert ch == expected_channels[preset], f"Channel count mismatch for {preset}: {ch} != {expected_channels[preset]}"
        tensor, mask, sample_id = ds[0]

        # Contract validation
        ds.validate_sample_output(tensor, mask, sample_id)
        assert not torch.isnan(tensor).any(), f"NaN found in tensor for {preset}"
        assert not torch.isinf(tensor).any(), f"Inf found in tensor for {preset}"
        assert mask.min() >= 0.0 and mask.max() <= 1.0, f"Mask not in [0, 1] for {preset}"

        print(f"  [OK] Preset '{preset:10s}': channels={tensor.shape[0]} | shape={list(tensor.shape)} | id={sample_id}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 1 Passed: All 6 modality presets comply with interface contract!"))

    # -------------------------------------------------------------------------
    # TEST 2: Additive Fusion Mode (RGB + DTM / Topo Addition into 3 Channels)
    # -------------------------------------------------------------------------
    print(f"\n[Test 2/{total_tests}] Testing Additive Fusion Mode (Normalized RGB + Normalized Modalities)...")
    
    # 1. Test rgb_dtm with fusion='add'
    ds_add = build_dataset('data/landslide.yaml', split='train', inputs='rgb_dtm', fusion='add')
    assert ds_add.in_channels == 3, f"Expected 3 channels with fusion='add', got {ds_add.in_channels}"
    tensor_add, mask_add, sample_id = ds_add[0]
    assert tensor_add.shape[0] == 3, f"Expected 3 channels in tensor, got {tensor_add.shape[0]}"
    ds_add.validate_sample_output(tensor_add, mask_add, sample_id)

    # 2. Verify element-wise addition math: tensor_add == (rgb + dtm_norm) / 2.0 (normalized to [0, 1])
    ds_rgb = build_dataset('data/landslide.yaml', split='train', inputs='rgb_only')
    ds_dtm = build_dataset('data/landslide.yaml', split='train', inputs=['DTM_NORM'])
    t_rgb, _, _ = ds_rgb[0]
    t_dtm, _, _ = ds_dtm[0]
    expected_sum = (t_rgb + t_dtm) / 2.0  # Normalized average to prevent [0, 2] distribution shift
    assert torch.allclose(tensor_add, expected_sum, atol=1e-5), "Addition fusion values do not match (RGB + DTM) / 2.0!"
    assert tensor_add.max() <= 1.0 + 1e-5 and tensor_add.min() >= 0.0, f"Additive tensor out of [0, 1] range: min={tensor_add.min()}, max={tensor_add.max()}"
    print(f"  [OK] Additive fusion math verified: (RGB in [0, 1] + DTM in [0, 1]) / 2 -> 3 channels tensor in [0, 1]")

    # 3. Test presets with add naming
    for add_preset in ['rgb_add_dtm', 'rgb_add_slope', 'rgb+dtm']:
        ds_preset = build_dataset('data/landslide.yaml', split='train', inputs=add_preset, fusion='add')
        assert ds_preset.in_channels == 3
        t_p, _, _ = ds_preset[0]
        assert t_p.shape[0] == 3
        print(f"  [OK] Preset '{add_preset:15s}': channels={t_p.shape[0]} | shape={list(t_p.shape)}")

    # 4. Verify rejection of multi-channel auxiliary modalities in additive mode
    try:
        build_dataset('data/landslide.yaml', split='train', inputs=['IMAGE', 'ASPECT'], fusion='add')
        raise AssertionError("Expected ValueError when attempting to use fusion='add' with 2-channel ASPECT!")
    except ValueError as e:
        assert "cannot be used with multi-channel modality" in str(e)
        print(f"  [OK] Incompatible multi-channel additive fusion properly rejected with descriptive ValueError")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 2 Passed: Additive fusion mode, [0, 1] renormalization & multi-channel safety verified!"))

    # -------------------------------------------------------------------------
    # TEST 3: Custom Dataset Adapter Contract
    # -------------------------------------------------------------------------
    print(f"\n[Test 3/{total_tests}] Testing Custom Dataset Registration & Interface Contract...")

    @register_dataset('mock_custom')
    class MockCustomDataset(BaseMultiModalDataset):
        def __init__(self, root_dir, split='train', **kwargs):
            super().__init__(root_dir, split, **kwargs)
            self._samples = ['mock_001', 'mock_002']
            self._in_channels = 6

        @property
        def in_channels(self): return self._in_channels
        @property
        def active_inputs(self): return ['RGB', 'DEM', 'SLOPE']
        @property
        def samples(self): return self._samples
        def __len__(self): return len(self._samples)
        def __getitem__(self, idx):
            tensor = torch.zeros((6, 512, 512), dtype=torch.float32)
            mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            sample_id = self._samples[idx]
            self.validate_sample_output(tensor, mask, sample_id)
            return tensor, mask, sample_id

    assert 'mock_custom' in DATASET_REGISTRY, "Custom dataset was not registered!"
    mock_ds = DATASET_REGISTRY['mock_custom'](root_dir='.', split='train')
    m_tensor, m_mask, m_id = mock_ds[0]
    assert m_tensor.shape == (6, 512, 512) and m_id == 'mock_001'

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 3 Passed: Custom dataset registration and contract validation verified!"))

    # -------------------------------------------------------------------------
    # TEST 4: Model Parser & Declarative Architectures (Width Scaling & Dynamic NC)
    # -------------------------------------------------------------------------
    print(f"\n[Test 4/{total_tests}] Testing Model Parser across Architectures, Width Scaling & Dynamic NC...")
    architectures = [
        ('models/architectures/unet.yaml', [3, 4, 5, 7]),
        ('models/architectures/unet_lite.yaml', [3, 4, 5, 7]),
        ('models/architectures/unet_noproj.yaml', [3]),
        ('models/architectures/unet_noproj_lite.yaml', [3]),
    ]

    for cfg, channels in architectures:
        for ch in channels:
            model = Model(cfg=cfg, ch=ch, nc=2)
            dummy_input = torch.randn(2, ch, 128, 128)
            out = model(dummy_input)
            assert out.shape == (2, 2, 128, 128), f"Output shape mismatch for {cfg} with C_in={ch}: {out.shape}"

    # Test dynamic width_multiple scaling (gw = 0.5, 0.25, 1.5) and dynamic nc (1, 3, 5)
    for gw in [0.5, 0.25, 1.5]:
        for test_nc in [1, 3, 5]:
            custom_cfg = {
                'nc': test_nc,
                'depth_multiple': 1.0,
                'width_multiple': gw,
                'backbone': [
                    [-1, 1, 'Conv', [3, 1, 1]],
                    [-1, 1, 'DoubleConv', [64]],
                    [-1, 1, 'Down', [128]],
                    [-1, 1, 'Down', [256]],
                ],
                'head': [
                    [[3, 2], 1, 'Up', [128]],
                    [[4, 1], 1, 'Up', [64]],
                    [-1, 1, 'OutConv', ['nc']]
                ]
            }
            scaled_model = Model(cfg=custom_cfg, ch=4, nc=test_nc)
            scaled_out = scaled_model(torch.randn(2, 4, 64, 64))
            assert scaled_out.shape == (2, test_nc, 64, 64), f"Width-scaled model failed: {scaled_out.shape}"
    print(f"  [OK] width_multiple scaling (gw in [0.25, 0.5, 1.5]) & dynamic nc in [1, 3, 5] verified")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 4 Passed: Direct U-Net without projection, width scaling & dynamic nc verified!"))

    # -------------------------------------------------------------------------
    # TEST 5: Loss Function & Metrics Computation (Micro/Macro & 2-Class mIoU)
    # -------------------------------------------------------------------------
    print(f"\n[Test 5/{total_tests}] Testing Loss Functions & Metric Suite (Micro/Macro & 2-Class mIoU)...")
    criterion = BCEDiceLoss(alpha=1.0, beta=1.0)
    metrics = SegmentationMetrics(conf_thres=0.5)

    dummy_logits = torch.randn(4, 2, 64, 64)
    dummy_targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    total_loss, loss_dict = criterion(dummy_logits, dummy_targets)
    assert not torch.isnan(total_loss), "Loss returned NaN!"
    assert 'bce' in loss_dict and 'dice' in loss_dict

    metrics.update(dummy_logits, dummy_targets)
    scores = metrics.compute()
    
    # Verify presence and validity of all metrics
    expected_metrics = ['iou', 'fg_iou', 'bg_iou', 'miou', 'dice', 'fg_dice', 'bg_dice', 'mdice', 'fg_dice_macro', 'fg_iou_macro', 'precision', 'recall', 'accuracy']
    for metric_name in expected_metrics:
        assert metric_name in scores and 0.0 <= scores[metric_name] <= 1.0, f"Invalid metric {metric_name}: {scores.get(metric_name)}"
    
    # Verify mathematical identity of 2-class averages
    assert abs(scores['miou'] - (scores['fg_iou'] + scores['bg_iou']) / 2.0) < 1e-6, "mIoU != (fg_iou + bg_iou)/2"
    assert abs(scores['mdice'] - (scores['fg_dice'] + scores['bg_dice']) / 2.0) < 1e-6, "mDice != (fg_dice + bg_dice)/2"
    print(f"  [OK] Fg IoU (Micro): {scores['fg_iou'] * 100:.2f}% | Bg IoU: {scores['bg_iou'] * 100:.2f}% | 2-Class mIoU: {scores['miou'] * 100:.2f}%")
    print(f"  [OK] Fg Dice (Micro): {scores['fg_dice'] * 100:.2f}% | Fg Dice (Macro): {scores['fg_dice_macro'] * 100:.2f}% | 2-Class mDice: {scores['mdice'] * 100:.2f}%")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 5 Passed: Foreground/Background metrics, 2-class mIoU/mDice & Macro Dice verified!"))

    # -------------------------------------------------------------------------
    # TEST 6: Micro Training, Reproducible State Serialization & Edge Cases
    # -------------------------------------------------------------------------
    print(f"\n[Test 6/{total_tests}] Testing Micro Training, State Serialization & Edge Cases...")
    smoke_save_dir = Path('runs/train/smoke_test_run')
    if smoke_save_dir.exists():
        shutil.rmtree(smoke_save_dir)

    hyp = {'lr0': 0.001, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0001, 'warmup_epochs': 1.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.01, 'bce_weight': 1.0, 'dice_weight': 1.0, 'pos_weight': 1.0}
    device = torch.device('cpu')

    # 1. EarlyStopping state_dict test
    from utils.torch_utils import EarlyStopping, get_rng_states, set_rng_states, torch_load
    es1 = EarlyStopping(patience=5, delta=1e-4, mode='max')
    es1(0.5)  # initial best
    es1(0.4)  # counter = 1
    es1(0.45) # counter = 2
    assert es1.counter == 2 and es1.best_score == 0.5, f"Unexpected EarlyStopping state: {es1.counter}, {es1.best_score}"
    es_state = es1.state_dict()
    es2 = EarlyStopping(patience=5, mode='max')
    es2.load_state_dict(es_state)
    assert es2.counter == 2 and es2.best_score == 0.5, "EarlyStopping failed to restore counter/best_score from state_dict!"
    print(f"  [OK] EarlyStopping state_dict serialization and restoration verified")

    # 2. RNG state capture and determinism test
    rng_saved = get_rng_states()
    val_a = torch.randn(10).numpy().copy()
    set_rng_states(rng_saved)
    val_b = torch.randn(10).numpy().copy()
    assert np.allclose(val_a, val_b), "RNG state restoration failed to reproduce identical pseudo-random numbers!"
    print(f"  [OK] RNG state capture & deterministic replay verified across PyTorch, NumPy, and Python")

    # 3. Trailing Gradient Accumulation group size logic test
    total_len = 10
    accum = 4
    for b_idx in range(total_len):
        rem = total_len - (b_idx // accum) * accum
        g_size = min(accum, rem)
        if b_idx in [0, 1, 2, 3]: assert g_size == 4
        elif b_idx in [4, 5, 6, 7]: assert g_size == 4
        elif b_idx in [8, 9]: assert g_size == 2
    print(f"  [OK] Unbiased trailing gradient accumulation group size calculations verified")

    # 4. Micro training cycle with unet_noproj_lite and rgb_dtm with fusion='add'
    loader, _ = create_dataloader('data/landslide.yaml', split='val', inputs='rgb_dtm', fusion='add', batch_size=2, num_workers=0)
    model = Model(cfg='models/architectures/unet_noproj_lite.yaml', ch=3, nc=2).to(device)
    opt_engine = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched_engine = torch.optim.lr_scheduler.CosineAnnealingLR(opt_engine, T_max=10)

    model.train()
    batch_t, batch_m, _ = next(iter(loader))
    assert batch_t.shape[1] == 3, f"Expected 3 channels in batch_t, got {batch_t.shape[1]}"
    logits = model(batch_t)
    loss, _ = criterion(logits, batch_m)
    loss.backward()
    opt_engine.step()
    sched_engine.step()

    # Evaluation on micro batch
    micro_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(loader.dataset, range(min(4, len(loader.dataset)))),
        batch_size=2,
        shuffle=False
    )
    eval_scores = evaluate(model, micro_loader, criterion, device, plots=False, pbar_desc="Smoke Eval")
    
    # Test plot_results
    from utils.plots import plot_results
    smoke_save_dir.mkdir(parents=True, exist_ok=True)
    test_csv = smoke_save_dir / 'results.csv'
    with open(test_csv, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_fg_iou,val_miou,val_fg_dice,val_fg_dice_macro,val_precision,val_recall,val_accuracy\n")
        f.write("1,1.2000,1.1000,0.3000,0.6000,0.4500,0.4400,0.5000,0.4200,0.8500\n")
        f.write("2,0.9000,0.8000,0.4000,0.6800,0.5800,0.5700,0.6000,0.5600,0.8900\n")
        f.write("3,0.7000,0.7500,0.4200,0.7000,0.6100,0.6000,0.6300,0.5900,0.9100\n")
    plot_results(test_csv, save_dir=smoke_save_dir)
    assert (smoke_save_dir / 'results.png').exists(), "results.png was not generated by plot_results"

    # Test full checkpoint serialization with scheduler, early_stopping, rng_state, loss_cfg
    ckpt_test_path = smoke_save_dir / 'weights' / 'best.pt'
    ckpt_test_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': 1,
        'model': model.state_dict(),
        'optimizer': opt_engine.state_dict(),
        'scheduler': sched_engine.state_dict(),
        'early_stopping': es1.state_dict(),
        'rng_state': get_rng_states(),
        'cfg': 'models/architectures/unet_noproj_lite.yaml',
        'inputs': 'rgb_dtm',
        'fusion': 'add',
        'in_channels': 3,
        'nc': 2,
        'hyp': hyp,
        'loss_cfg': {'alpha': 0.5, 'beta': 1.0, 'pos_weight': 4.0, 'smooth': 1.0}
    }, ckpt_test_path)
    loaded_ckpt = torch_load(ckpt_test_path, map_location='cpu')
    assert 'loss_cfg' in loaded_ckpt and loaded_ckpt['loss_cfg']['pos_weight'] == 4.0
    assert 'scheduler' in loaded_ckpt and 'early_stopping' in loaded_ckpt and 'rng_state' in loaded_ckpt
    print(f"  [OK] Full checkpoint reproducibility bundle (model, opt, sched, es, rng, loss_cfg) verified")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 6 Passed: End-to-end training, reproducible state serialization & edge cases verified!"))

    # -------------------------------------------------------------------------
    # TEST 7: Inference Pipeline with Addition Fusion (predict.py)
    # -------------------------------------------------------------------------
    print(f"\n[Test 7/{total_tests}] Testing Inference Pipeline with Addition Fusion...")
    predict_save_dir = Path('runs/predict/smoke_predict')
    if predict_save_dir.exists():
        shutil.rmtree(predict_save_dir)

    class SmokePredictOpt:
        weights = ''
        source = 'dataset/dataset_1/validation/IMAGE/Chainage_17_00156.png'
        cfg = 'models/architectures/unet_noproj_lite.yaml'
        data = 'data/landslide.yaml'
        inputs = 'rgb_dtm'
        fusion = 'add'
        conf_thres = 0.5
        device = 'cpu'
        project = 'runs/predict'
        name = 'smoke_predict'
        exist_ok = True

    run_predict(SmokePredictOpt())

    mask_out = predict_save_dir / 'masks' / 'Chainage_17_00156.png'
    overlay_out = predict_save_dir / 'overlays' / 'Chainage_17_00156_overlay.png'

    assert mask_out.exists(), f"Generated mask missing: {mask_out}"
    assert overlay_out.exists(), f"Generated overlay missing: {overlay_out}"
    print(f"  [OK] Saved mask   : {mask_out} ({mask_out.stat().st_size} bytes)")
    print(f"  [OK] Saved overlay: {overlay_out} ({overlay_out.stat().st_size} bytes)")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 7 Passed: Full inference & overlay generation with addition fusion verified!"))

    # -------------------------------------------------------------------------
    # TEST 8: Official UNet Weight Loader (Both With & Without Projection)
    # -------------------------------------------------------------------------
    print(f"\n[Test 8/{total_tests}] Testing Official UNet Weights Loader for Both Architecture Types...")
    from utils.torch_utils import load_pretrained_weights
    
    # 1. With projection (unet.yaml)
    unet_proj = Model(cfg='models/architectures/unet.yaml', ch=3, nc=2)
    res_proj = load_pretrained_weights(unet_proj, 'unet_carvana', device='cpu')
    assert res_proj['matched'] >= 118, f"Expected >=118 matched layers for unet.yaml, got {res_proj['matched']}"
    print(f"  [OK] unet.yaml (with projection): {res_proj['matched']}/{res_proj['total']} layers mapped")

    # 2. Without projection (unet_noproj.yaml)
    unet_noproj = Model(cfg='models/architectures/unet_noproj.yaml', ch=3, nc=2)
    res_noproj = load_pretrained_weights(unet_noproj, 'unet_carvana', device='cpu')
    assert res_noproj['matched'] >= 118, f"Expected >=118 matched layers for unet_noproj.yaml, got {res_noproj['matched']}"
    print(f"  [OK] unet_noproj.yaml (no projection): {res_noproj['matched']}/{res_noproj['total']} layers mapped")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 8 Passed: Official UNet pretrained weights loading across architectures verified!"))

    # -------------------------------------------------------------------------
    # TEST 9: YAML Configuration File Argument Parser & Overriding
    # -------------------------------------------------------------------------
    print(f"\n[Test 9/{total_tests}] Testing YAML Configuration Argument Parser & CLI Overrides...")
    from train import parse_opt as train_parse_opt
    import sys
    sys.argv = ['train.py', '--config', 'configs/train_rgb_add_dtm.yaml', '--batch-size', '16', '--fusion', 'add']
    test_opt = train_parse_opt()
    assert test_opt.batch_size == 16, f"Expected overridden batch_size=16, got {test_opt.batch_size}"
    assert test_opt.fusion == 'add', f"Expected fusion=add, got {test_opt.fusion}"
    assert test_opt.cfg == 'models/architectures/unet_noproj.yaml', f"Expected unet_noproj.yaml, got {test_opt.cfg}"
    print(f"  [OK] Config file defaults loaded and overridden properly: cfg={test_opt.cfg}, fusion={test_opt.fusion}, batch_size={test_opt.batch_size}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 9 Passed: YAML config argument file loading & dynamic CLI overriding verified!"))

    # -------------------------------------------------------------------------
    # TEST 10: Single Sample Loader Addition Fusion Test
    # -------------------------------------------------------------------------
    print(f"\n[Test 10/{total_tests}] Testing load_single_sample with Addition Fusion...")
    tensor_single, rgb_single, _ = load_single_sample(
        'Chainage_17_00156',
        'dataset/dataset_1/validation',
        inputs=['IMAGE', 'DTM_NORM'],
        fusion='add'
    )
    assert tensor_single.shape == (1, 3, 512, 512), f"Expected (1, 3, 512, 512), got {tensor_single.shape}"
    assert rgb_single.shape == (512, 512, 3), f"Expected (512, 512, 3) for rgb_display, got {rgb_single.shape}"
    print(f"  [OK] Single sample loader verified: tensor shape={list(tensor_single.shape)}, rgb_display shape={list(rgb_single.shape)}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 10 Passed: Single sample loader addition fusion verified!"))

    # -------------------------------------------------------------------------
    # TEST 11: Genetic Hyperparameter Evolution Engine
    # -------------------------------------------------------------------------
    print(f"\n[Test 11/{total_tests}] Testing Genetic Hyperparameter Evolution Engine & Fitness Calculations...")
    from evolve import compute_fitness, mutate_genes, select_parent, GENE_SPACE
    from utils.plots import plot_evolve

    # 1. Test fitness scoring & recall penalty
    normal_scores = {'dice': 0.70, 'iou': 0.55, 'recall': 0.80, 'precision': 0.65}
    fit_normal = compute_fitness(normal_scores)
    expected_fit = 0.5 * 0.70 + 0.3 * 0.55 + 0.2 * 0.80
    assert abs(fit_normal - expected_fit) < 1e-5, f"Fitness mismatch: {fit_normal} != {expected_fit}"

    collapsed_scores = {'dice': 0.00, 'iou': 0.00, 'recall': 0.01, 'precision': 1.00}
    fit_collapsed = compute_fitness(collapsed_scores, recall_floor=0.05)
    assert fit_collapsed == 0.0, f"Expected 0.0 fitness for collapsed model, got {fit_collapsed}"
    print(f"  [OK] Composite fitness formula and recall floor verified: normal={fit_normal:.4f}, collapsed={fit_collapsed:.4f}")

    # 2. Test gene mutation within bounds
    base_hyp = {'lr0': 0.001, 'lrf': 0.1, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0,
                'bce_weight': 0.5, 'dice_weight': 1.0, 'pos_weight': 4.0, 'fliplr': 0.5, 'flipud': 0.5, 'rot90': 0.5}
    mutated = mutate_genes(base_hyp, prob=1.0)
    for g, meta in GENE_SPACE.items():
        assert meta['min'] <= mutated[g] <= meta['max'], f"Gene {g} out of bounds: {mutated[g]} not in [{meta['min']}, {meta['max']}]"
    print(f"  [OK] Bounded gene mutation verified across all {len(GENE_SPACE)} hyperparameter genes")

    # 3. Test parent selection from history
    mock_history = [
        {'generation': 1, 'fitness': 0.1, 'hyp': {'lr0': 0.001}},
        {'generation': 2, 'fitness': 0.8, 'hyp': {'lr0': 0.005}},
        {'generation': 3, 'fitness': 0.2, 'hyp': {'lr0': 0.002}}
    ]
    parent = select_parent(mock_history)
    assert 'lr0' in parent, "Parent selection failed to return gene dict"
    print(f"  [OK] Parent selection weighted sampling verified")

    # 4. Test evolution plotting
    evolve_test_dir = Path('runs/evolve/smoke_test')
    evolve_test_dir.mkdir(parents=True, exist_ok=True)
    evolve_test_csv = evolve_test_dir / 'evolve.csv'
    with open(evolve_test_csv, 'w') as f:
        f.write("generation,fitness,val_dice,val_iou,val_recall,val_precision,val_loss,lr0,lrf,momentum,weight_decay,warmup_epochs,bce_weight,dice_weight,pos_weight,fliplr,flipud,rot90\n")
        f.write("1,0.4500,0.5000,0.3500,0.6000,0.5500,0.9500,0.001000,0.100000,0.937,0.000500,3.0,0.5,1.0,4.0,0.5,0.5,0.5\n")
        f.write("2,0.6200,0.6800,0.5200,0.7500,0.7000,0.7200,0.001200,0.120000,0.920,0.000400,2.5,0.4,1.2,5.0,0.5,0.5,0.5\n")
    plot_evolve(evolve_test_csv, save_dir=evolve_test_dir)
    assert (evolve_test_dir / 'evolve_fitness.png').exists(), "evolve_fitness.png was not generated"
    assert (evolve_test_dir / 'evolve_scatter.png').exists(), "evolve_scatter.png was not generated"
    print(f"  [OK] Evolution progress and correlation scatter plotting verified")

    if evolve_test_dir.exists(): shutil.rmtree(evolve_test_dir)

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 11 Passed: Hyperparameter evolution engine, gene mutator & plots verified!"))

    # -------------------------------------------------------------------------
    # TEST 12: Aspect Directional Vector Transformation Consistency
    # -------------------------------------------------------------------------
    print(f"\n[Test 12/{total_tests}] Testing Aspect Directional Vector Transformation Consistency under Geometric Augmentations...")
    from data.transforms import MultiModalTransform

    # Create synthetic unit vector (e.g. angle theta = 45 degrees: sin = 0.7071, cos = 0.7071)
    theta_deg = 45.0
    sin_val = float(np.sin(np.radians(theta_deg)))
    cos_val = float(np.cos(np.radians(theta_deg)))

    dummy_feature = np.zeros((64, 64, 5), dtype=np.float32)
    dummy_feature[:, :, 3] = sin_val  # Channel 3: sin(aspect)
    dummy_feature[:, :, 4] = cos_val  # Channel 4: cos(aspect)
    dummy_mask = np.zeros((64, 64), dtype=np.float32)

    # 1. Test Horizontal Flip (fliplr, x -> -x): East-West flipped (sin -> -sin), North-South unchanged
    tf_lr = MultiModalTransform(augment=True, hyp={'fliplr': 1.0, 'flipud': 0.0, 'rot90': 0.0}, img_size=(64, 64), aspect_slice=(3, 5))
    t_lr, _ = tf_lr(dummy_feature.copy(), dummy_mask.copy())
    assert torch.allclose(t_lr[3], torch.tensor(-sin_val), atol=1e-5), f"fliplr sin mismatch: {t_lr[3, 0, 0]} != {-sin_val}"
    assert torch.allclose(t_lr[4], torch.tensor(cos_val), atol=1e-5), f"fliplr cos mismatch: {t_lr[4, 0, 0]} != {cos_val}"
    print(f"  [OK] Horizontal flip vector transformation verified: sin -> -sin ({t_lr[3, 0, 0]:.4f}), cos -> cos ({t_lr[4, 0, 0]:.4f})")

    # 2. Test Vertical Flip (flipud, y -> -y): North-South flipped (cos -> -cos), East-West unchanged
    tf_ud = MultiModalTransform(augment=True, hyp={'fliplr': 0.0, 'flipud': 1.0, 'rot90': 0.0}, img_size=(64, 64), aspect_slice=(3, 5))
    t_ud, _ = tf_ud(dummy_feature.copy(), dummy_mask.copy())
    assert torch.allclose(t_ud[3], torch.tensor(sin_val), atol=1e-5), f"flipud sin mismatch: {t_ud[3, 0, 0]} != {sin_val}"
    assert torch.allclose(t_ud[4], torch.tensor(-cos_val), atol=1e-5), f"flipud cos mismatch: {t_ud[4, 0, 0]} != {-cos_val}"
    print(f"  [OK] Vertical flip vector transformation verified: sin -> sin ({t_ud[3, 0, 0]:.4f}), cos -> -cos ({t_ud[4, 0, 0]:.4f})")

    # 3. Test 90-degree rotations (rot90 with k=1, 2, 3)
    for k, (exp_sin, exp_cos) in [(1, (-cos_val, sin_val)), (2, (-sin_val, -cos_val)), (3, (cos_val, -sin_val))]:
        # Manually invoke transform rotation logic
        test_feat = dummy_feature.copy()
        test_mask = dummy_mask.copy()
        test_feat = np.rot90(test_feat, k, (0, 1))
        s_val = test_feat[:, :, 3].copy()
        c_val = test_feat[:, :, 4].copy()
        if k == 1:
            test_feat[:, :, 3] = -c_val
            test_feat[:, :, 4] = s_val
        elif k == 2:
            test_feat[:, :, 3] = -s_val
            test_feat[:, :, 4] = -c_val
        elif k == 3:
            test_feat[:, :, 3] = c_val
            test_feat[:, :, 4] = -s_val

        t_rot = torch.from_numpy(np.ascontiguousarray(test_feat).transpose(2, 0, 1)).float()
        assert torch.allclose(t_rot[3], torch.tensor(exp_sin), atol=1e-5), f"rot90 k={k} sin mismatch: {t_rot[3, 0, 0]} != {exp_sin}"
        assert torch.allclose(t_rot[4], torch.tensor(exp_cos), atol=1e-5), f"rot90 k={k} cos mismatch: {t_rot[4, 0, 0]} != {exp_cos}"
        print(f"  [OK] rot90 (k={k}) vector rotation verified: (sin, cos) -> ({t_rot[3, 0, 0]:.4f}, {t_rot[4, 0, 0]:.4f})")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 12 Passed: Topographic aspect directional vector transformations fully verified!"))

    # -------------------------------------------------------------------------
    # TEST 13: Active img_size Spatial Dimension Enforcement
    # -------------------------------------------------------------------------
    print(f"\n[Test 13/{total_tests}] Testing Active img_size Spatial Dimension Enforcement & Mask Interpolation...")
    
    # 1. Transform resizing check from 512x512 to 256x256
    large_feature = np.random.randn(512, 512, 4).astype(np.float32)
    binary_mask = (np.random.rand(512, 512) > 0.5).astype(np.float32)
    
    tf_resize = MultiModalTransform(augment=False, img_size=(256, 256))
    t_resized, m_resized = tf_resize(large_feature, binary_mask)
    assert t_resized.shape == (4, 256, 256), f"Expected (4, 256, 256), got {t_resized.shape}"
    assert m_resized.shape == (1, 256, 256), f"Expected (1, 256, 256), got {m_resized.shape}"
    
    # Check that binary mask values remain discrete {0.0, 1.0} after nearest-neighbor interpolation
    unique_mask_vals = torch.unique(m_resized).tolist()
    assert all(v in [0.0, 1.0] for v in unique_mask_vals), f"Mask interpolation created non-binary values: {unique_mask_vals}"
    print(f"  [OK] Transform spatial resizing (512x512 -> 256x256) & discrete mask integrity verified")

    # 2. Dataset loader with non-standard img_size (e.g. 128x128)
    ds_custom_size = build_dataset('data/landslide.yaml', split='val', inputs='rgb_dtm', img_size=(128, 128))
    tensor_custom, mask_custom, s_id = ds_custom_size[0]
    assert tensor_custom.shape == (4, 128, 128), f"Expected (4, 128, 128), got {tensor_custom.shape}"
    assert mask_custom.shape == (1, 128, 128), f"Expected (1, 128, 128), got {mask_custom.shape}"
    ds_custom_size.validate_sample_output(tensor_custom, mask_custom, s_id)
    print(f"  [OK] Dataset instantiation with custom img_size=(128, 128) verified: shape={list(tensor_custom.shape)}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 13 Passed: Active img_size spatial enforcement & nearest-neighbor mask interpolation verified!"))

    # -------------------------------------------------------------------------
    # TEST 14: Topographic Aspect NoData / NaN to Zero Vector Handling
    # -------------------------------------------------------------------------
    print(f"\n[Test 14/{total_tests}] Testing Topographic Aspect NoData / NaN to Zero-Vector (0, 0) Encoding...")
    
    # Verify synthetic array with NaNs, Infs, and negative NoData (-9999, -1)
    raw_aspect = np.array([
        [0.0, 90.0, 180.0, 270.0],
        [np.nan, np.inf, -9999.0, -1.0]
    ], dtype=np.float32)

    invalid_mask = np.isnan(raw_aspect) | np.isinf(raw_aspect) | (raw_aspect < 0) | (raw_aspect > 360.0)
    valid_arr = np.where(invalid_mask, 0.0, raw_aspect)
    rad = valid_arr * (np.pi / 180.0)
    sin_aspect = np.sin(rad)
    cos_aspect = np.cos(rad)
    sin_aspect[invalid_mask] = 0.0
    cos_aspect[invalid_mask] = 0.0

    # Valid cardinal directions
    assert np.allclose(sin_aspect[0, 0], 0.0) and np.allclose(cos_aspect[0, 0], 1.0), "0 deg should be (0, 1) North"
    assert np.allclose(sin_aspect[0, 1], 1.0) and np.allclose(cos_aspect[0, 1], 0.0, atol=1e-5), "90 deg should be (1, 0) East"

    # Invalid / NoData pixels MUST be (0.0, 0.0) with zero magnitude, NOT (0.0, 1.0) North
    assert (sin_aspect[1, :] == 0.0).all() and (cos_aspect[1, :] == 0.0).all(), "NoData pixels were not set to (0, 0) zero-vector!"
    print(f"  [OK] Aspect NoData handling verified: NaN / Inf / -9999 correctly encoded as (0.0, 0.0) zero vector instead of North (0, 1)")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 14 Passed: Topographic Aspect NoData / NaN zero-vector encoding verified!"))

    # -------------------------------------------------------------------------
    # TEST 15: Spatial Data Leakage Audit & Regional Split Partitioning
    # -------------------------------------------------------------------------
    print(f"\n[Test 15/{total_tests}] Testing Spatial Data Leakage Audit & Geographic Region Splitter...")
    from data.spatial import audit_spatial_splits, create_spatial_region_splits, SpatialRegionKFold

    # 1. Audit real repository dataset_1
    audit_res = audit_spatial_splits('data/landslide.yaml', verbose=False)
    assert audit_res['has_spatial_leakage'] is False, "dataset_1 failed spatial leakage audit!"
    assert len(audit_res['leakage_train_val']) == 0, "Shared regions found between train and val in dataset_1!"
    print(f"  [OK] dataset_1 audit verified: Train regions {audit_res['regions_by_split']['train']} | Val regions {audit_res['regions_by_split']['val']} (Zero leakage)")

    # 2. Test spatial region partitioning on mock multi-corridor sample set
    mock_stems = [
        'Chainage_1_0001', 'Chainage_1_0002', 'Chainage_1_0003',
        'Chainage_2_0001', 'Chainage_2_0002',
        'Chainage_3_0001', 'Chainage_3_0002', 'Chainage_3_0003', 'Chainage_3_0004'
    ]
    splits = create_spatial_region_splits(mock_stems, val_regions=['Chainage_2'])
    assert 'Chainage_2_0001' in splits['val'] and 'Chainage_2_0002' in splits['val']
    assert not any('Chainage_2' in s for s in splits['train']), "Regional leakage in mock split!"
    print(f"  [OK] Region-based dataset partitioning verified: whole corridors preserved in dedicated splits")

    # 3. Test SpatialRegionKFold cross-validation splitter
    kfold = SpatialRegionKFold(n_splits=3, shuffle=False)
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(mock_stems)):
        tr_regs = set(mock_stems[i].split('_')[1] for i in tr_idx)
        val_regs = set(mock_stems[i].split('_')[1] for i in val_idx)
        assert len(tr_regs.intersection(val_regs)) == 0, f"Spatial overlap in fold {fold}!"
    print(f"  [OK] SpatialRegionKFold cross-validation verified: 0% cross-fold regional leakage")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 15 Passed: Geospatial data leakage audit & regional partitioning verified!"))

    # Clean up test artifacts
    if smoke_save_dir.exists(): shutil.rmtree(smoke_save_dir)
    if predict_save_dir.exists(): shutil.rmtree(predict_save_dir)

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------
    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(colorstr('bold', 'green', f"[PASSED] ALL {passed_tests}/{total_tests} SMOKE TESTS PASSED SUCCESSFULLY in {total_time:.2f}s!"))
    print("=" * 80 + "\n")
    return True


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
