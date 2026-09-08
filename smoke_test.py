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
from predict import run_predict, parse_opt as parse_predict_opt


def run_smoke_test():
    print("=" * 80)
    print(colorstr('bold', 'cyan', "[SMOKE TEST] RUNNING COMPREHENSIVE REPOSITORY SMOKE TEST"))
    print("=" * 80)
    passed_tests = 0
    total_tests = 8
    t0 = time.time()

    # -------------------------------------------------------------------------
    # TEST 1: Dataset Registry & Adapter Presets
    # -------------------------------------------------------------------------
    print(f"\n[Test 1/{total_tests}] Testing Dataset Registry & All Modality Presets...")
    presets = ['rgb_only', 'topo_only', 'rgb_dtm', 'rgb_slope', 'rgb_aspect', 'all']
    expected_channels = {'rgb_only': 3, 'topo_only': 4, 'rgb_dtm': 4, 'rgb_slope': 4, 'rgb_aspect': 5, 'all': 7}

    for preset in presets:
        ds = build_dataset('data/landslide.yaml', split='train', inputs=preset)
        ch = get_channel_count(preset)
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
    # TEST 2: Custom Dataset Adapter Contract
    # -------------------------------------------------------------------------
    print(f"\n[Test 2/{total_tests}] Testing Custom Dataset Registration & Interface Contract...")

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
    print(colorstr('bright_green', f"--> Test 2 Passed: Custom dataset registration and contract validation verified!"))

    # -------------------------------------------------------------------------
    # TEST 3: Model Parser & Declarative U-Net Architectures
    # -------------------------------------------------------------------------
    print(f"\n[Test 3/{total_tests}] Testing Model Parser & Forward Pass across Architectures...")
    for cfg in ['models/architectures/unet.yaml', 'models/architectures/unet_lite.yaml']:
        for ch in [3, 4, 5, 7]:
            model = Model(cfg=cfg, ch=ch, nc=1)
            dummy_input = torch.randn(2, ch, 128, 128)
            out = model(dummy_input)
            assert out.shape == (2, 1, 128, 128), f"Output shape mismatch for {cfg}: {out.shape}"
        print(f"  [OK] Model '{cfg}': verified forward pass for C_in in [3, 4, 5, 7]")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 3 Passed: Dynamic Conv input projection & U-Net architectures verified!"))

    # -------------------------------------------------------------------------
    # TEST 4: Loss Function & Metrics Computation
    # -------------------------------------------------------------------------
    print(f"\n[Test 4/{total_tests}] Testing Loss Functions & Metric Suite...")
    criterion = BCEDiceLoss(alpha=1.0, beta=1.0)
    metrics = SegmentationMetrics(conf_thres=0.5)

    dummy_logits = torch.randn(4, 1, 64, 64)
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
    print(colorstr('bright_green', f"--> Test 4 Passed: Combined BCE + Dice loss & Metric accumulator verified!"))

    # -------------------------------------------------------------------------
    # TEST 5: End-to-End Micro Training Cycle
    # -------------------------------------------------------------------------
    print(f"\n[Test 5/{total_tests}] Testing End-to-End Micro Training Cycle...")
    smoke_save_dir = Path('runs/train/smoke_test_run')
    if smoke_save_dir.exists():
        shutil.rmtree(smoke_save_dir)

    # Micro training test: 1 epoch on validation set (small)
    class SmokeOpt:
        weights = ''
        cfg = 'models/architectures/unet_lite.yaml'
        data = 'data/landslide.yaml'
        hyp = 'data/hyp.scratch.yaml'
        inputs = 'rgb_dtm'
        epochs = 1
        batch_size = 4
        img_size = 512
        conf_thres = 0.5
        device = 'cpu'
        workers = 0
        project = 'runs/train'
        name = 'smoke_test_run'
        exist_ok = True
        patience = 5
        seed = 42

    hyp = {'lr0': 0.001, 'lrf': 0.01, 'weight_decay': 0.0001, 'bce_weight': 1.0, 'dice_weight': 1.0, 'pos_weight': 1.0}
    device = torch.device('cpu')

    # Run micro-epoch
    loader, _ = create_dataloader('data/landslide.yaml', split='val', inputs='rgb_dtm', batch_size=2, num_workers=0)
    model = Model(cfg=SmokeOpt.cfg, ch=4, nc=1).to(device)
    opt_engine = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    batch_t, batch_m, _ = next(iter(loader))
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
    assert 'iou' in eval_scores and 'dice' in eval_scores
    print(f"  [OK] Micro training & validation passed: Loss={eval_scores['loss']:.4f}, Dice={eval_scores['dice']*100:.2f}%")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 5 Passed: End-to-end forward/backward/eval cycle verified!"))

    # -------------------------------------------------------------------------
    # TEST 6: Inference Pipeline (predict.py)
    # -------------------------------------------------------------------------
    print(f"\n[Test 6/{total_tests}] Testing Inference & Visual Overlay Generator...")
    predict_save_dir = Path('runs/predict/smoke_predict')
    if predict_save_dir.exists():
        shutil.rmtree(predict_save_dir)

    class SmokePredictOpt:
        weights = ''
        source = 'dataset/dataset_1/validation/IMAGE/Chainage_17_00156.png'
        cfg = 'models/architectures/unet_lite.yaml'
        data = 'data/landslide.yaml'
        inputs = 'rgb_aspect'
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
    print(colorstr('bright_green', f"--> Test 6 Passed: Full inference & overlay generation verified!"))

    # -------------------------------------------------------------------------
    # TEST 7: Official UNet Weight Loader & Key Remapping
    # -------------------------------------------------------------------------
    print(f"\n[Test 7/{total_tests}] Testing Official UNet Weights Loader & Key Remapping...")
    from utils.torch_utils import load_pretrained_weights
    unet_model = Model(cfg='models/architectures/unet.yaml', ch=3, nc=1)
    res = load_pretrained_weights(unet_model, 'unet_carvana', device='cpu')
    assert res['matched'] > 100, f"Expected >100 matched layers, got {res['matched']}"
    print(f"  [OK] Official UNet Carvana weights mapped: {res['matched']}/{res['total']} layers into declarative Model")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 7 Passed: Official UNet pretrained weights loading & layer remapping verified!"))

    # -------------------------------------------------------------------------
    # TEST 8: YAML Configuration File Argument Parser & Overriding
    # -------------------------------------------------------------------------
    print(f"\n[Test 8/{total_tests}] Testing YAML Configuration Argument Parser & CLI Overrides...")
    from train import parse_opt as train_parse_opt
    import sys
    sys.argv = ['train.py', '--config', 'configs/train.yaml', '--batch-size', '16', '--inputs', 'all']
    test_opt = train_parse_opt()
    assert test_opt.batch_size == 16, f"Expected overridden batch_size=16, got {test_opt.batch_size}"
    assert test_opt.inputs == 'all', f"Expected overridden inputs=all, got {test_opt.inputs}"
    assert test_opt.cache_ram == True, f"Expected config default cache_ram=True, got {test_opt.cache_ram}"
    print(f"  [OK] Config file defaults loaded and overridden properly: batch_size={test_opt.batch_size}, inputs={test_opt.inputs}")

    passed_tests += 1
    print(colorstr('bright_green', f"--> Test 8 Passed: YAML config argument file loading & dynamic CLI overriding verified!"))

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
