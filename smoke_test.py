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
    total_tests = 11
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

    # 2. Verify element-wise addition math: tensor_add == rgb + dtm_norm
    ds_rgb = build_dataset('data/landslide.yaml', split='train', inputs='rgb_only')
    ds_dtm = build_dataset('data/landslide.yaml', split='train', inputs=['DTM_NORM'])
    t_rgb, _, _ = ds_rgb[0]
    t_dtm, _, _ = ds_dtm[0]
    expected_sum = t_rgb + t_dtm  # Broadcast 1ch DTM across 3 RGB channels
    assert torch.allclose(tensor_add, expected_sum, atol=1e-5), "Addition fusion values do not match (RGB + DTM)!"
    print(f"  [OK] Additive fusion math verified: RGB in [0, 1] + DTM in [0, 1] -> 3 channels tensor")

    # 3. Test presets with add naming
    for add_preset in ['rgb_add_dtm', 'rgb_add_slope', 'rgb+dtm']:
        ds_preset = build_dataset('data/landslide.yaml', split='train', inputs=add_preset, fusion='add')
        assert ds_preset.in_channels == 3
        t_p, _, _ = ds_preset[0]
        assert t_p.shape[0] == 3
        print(f"  [OK] Preset '{add_preset:15s}': channels={t_p.shape[0]} | shape={list(t_p.shape)}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 2 Passed: Additive fusion mode & element-wise addition verified!"))

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
    # TEST 4: Model Parser & Declarative Architectures (With & Without Projection)
    # -------------------------------------------------------------------------
    print(f"\n[Test 4/{total_tests}] Testing Model Parser across All Architectures (With & Without Projection)...")
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
        print(f"  [OK] Model '{cfg}': verified forward pass for C_in in {channels}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 4 Passed: Direct U-Net without projection & standard U-Net forward passes verified!"))

    # -------------------------------------------------------------------------
    # TEST 5: Loss Function & Metrics Computation
    # -------------------------------------------------------------------------
    print(f"\n[Test 5/{total_tests}] Testing Loss Functions & Metric Suite...")
    criterion = BCEDiceLoss(alpha=1.0, beta=1.0)
    metrics = SegmentationMetrics(conf_thres=0.5)

    dummy_logits = torch.randn(4, 2, 64, 64)
    dummy_targets = torch.randint(0, 2, (4, 1, 64, 64)).float()

    total_loss, loss_dict = criterion(dummy_logits, dummy_targets)
    assert not torch.isnan(total_loss), "Loss returned NaN!"
    assert 'bce' in loss_dict and 'dice' in loss_dict

    metrics.update(dummy_logits, dummy_targets)
    scores = metrics.compute()
    for metric_name in ['iou', 'dice', 'precision', 'recall', 'accuracy']:
        assert metric_name in scores and 0.0 <= scores[metric_name] <= 1.0, f"Invalid metric {metric_name}: {scores.get(metric_name)}"
        print(f"  [OK] Metric {metric_name:<10s}: {scores[metric_name] * 100:.2f}%")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 5 Passed: Combined BCE + Dice loss & Metric accumulator verified!"))

    # -------------------------------------------------------------------------
    # TEST 6: End-to-End Micro Training Cycle (No-Projection + Addition Fusion)
    # -------------------------------------------------------------------------
    print(f"\n[Test 6/{total_tests}] Testing End-to-End Micro Training Cycle (No-Projection & Addition Fusion)...")
    smoke_save_dir = Path('runs/train/smoke_test_run')
    if smoke_save_dir.exists():
        shutil.rmtree(smoke_save_dir)

    hyp = {'lr0': 0.001, 'lrf': 0.01, 'weight_decay': 0.0001, 'bce_weight': 1.0, 'dice_weight': 1.0, 'pos_weight': 1.0}
    device = torch.device('cpu')

    # Run micro-epoch with unet_noproj_lite and rgb_dtm with fusion='add'
    loader, _ = create_dataloader('data/landslide.yaml', split='val', inputs='rgb_dtm', fusion='add', batch_size=2, num_workers=0)
    model = Model(cfg='models/architectures/unet_noproj_lite.yaml', ch=3, nc=2).to(device)
    opt_engine = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    batch_t, batch_m, _ = next(iter(loader))
    assert batch_t.shape[1] == 3, f"Expected 3 channels in batch_t, got {batch_t.shape[1]}"
    logits = model(batch_t)
    loss, _ = criterion(logits, batch_m)
    loss.backward()
    opt_engine.step()

    # Run evaluation on micro batch
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
        f.write("epoch,train_loss,val_loss,val_iou,val_dice,val_precision,val_recall,val_accuracy\n")
        f.write("1,1.2000,1.1000,0.3000,0.4500,0.5000,0.4200,0.8500\n")
        f.write("2,0.9000,0.8000,0.4000,0.5800,0.6000,0.5600,0.8900\n")
        f.write("3,0.7000,0.7500,0.4200,0.6100,0.6300,0.5900,0.9100\n")
    plot_results(test_csv, save_dir=smoke_save_dir)
    assert (smoke_save_dir / 'results.png').exists(), "results.png was not generated by plot_results"
    print(f"  [OK] Micro training & evaluation cycle completed with no-projection architecture & addition fusion")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 6 Passed: End-to-end forward/backward/eval cycle & metric plotting verified!"))

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
