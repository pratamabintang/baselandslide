# Multi-Modal Landslide Segmentation Baseline

A modular, declarative PyTorch repository for multi-modal landslide segmentation using U-Net architectures, declarative YAML layer parsing, dynamic $1\times1$ input projection, physical directional aspect transformations, and combined BCE + Dice loss.

---

## 📁 Repository Structure

```text
landslide/
├── dataset/                     # Local datasets (gitignored)
│   └── dataset_1/               # Primary multi-modal landslide dataset
│       ├── train/               # (IMAGE, LABEL, DTM, DTM_NORM, SLOPE, ASPECT) [910 samples]
│       ├── validation/          # (IMAGE, LABEL, DTM, DTM_NORM, SLOPE, ASPECT) [184 samples]
│       ├── global_raster_minmax.json
│       └── pairing_reports/
├── data/
│   ├── landslide.yaml           # Dataset configuration, paths, classes, presets
│   ├── hyp.scratch.yaml         # Training hyperparameters & loss weights
│   ├── base.py                  # BaseMultiModalDataset abstract interface & registry
│   ├── adapters.py              # FolderStructureAdapter & CustomDatasetWrapper
│   ├── transforms.py            # Synchronized spatial augmentations & Aspect vector rotation
│   ├── spatial.py               # Geospatial leakage audit & regional corridor partitioning
│   └── datasets.py              # Multi-modal dataset factory & loader
├── models/
│   ├── architectures/
│   │   ├── unet.yaml            # Standard U-Net with 1x1 Conv input projection
│   │   ├── unet_lite.yaml       # Lightweight U-Net with 1x1 Conv input projection
│   │   ├── unet_noproj.yaml     # Direct U-Net without projection (3-channel input)
│   │   └── unet_noproj_lite.yaml# Lightweight Direct U-Net without projection
│   ├── common.py                # Layer modules: Conv, DoubleConv, Down, Up, OutConv, Concat
│   └── model_parser.py          # Declarative YAML architecture parser & Model class
├── configs/                     # Pre-configured YAML experiment files
│   ├── train.yaml               # Standard training (RGB + DTM concatenation)
│   ├── train_rgb_noproj.yaml    # Pure 3-channel RGB direct baseline
│   ├── train_rgb_add_dtm.yaml   # RGB + DTM normalized additive fusion (3 channels)
│   ├── train_multimodal.yaml    # 7-channel multi-modal training
│   ├── train_finetune_carvana.yaml # Pretrained Carvana U-Net transfer learning
│   ├── test.yaml                # Standalone validation evaluation
│   ├── predict.yaml             # Multi-modal inference configuration
│   └── evolve.yaml              # Hyperparameter evolution configuration
├── utils/
│   ├── loss.py                  # Combined BCEWithLogitsLoss + Soft Dice Loss
│   ├── metrics.py               # Comprehensive metric suite (Fg/Bg IoU, 2-Class mIoU, Micro/Macro Dice)
│   ├── plots.py                 # Loss curves and multi-panel prediction overlays
│   ├── torch_utils.py           # Device management, seeding, EarlyStopping, and pretrained loader
│   └── general.py               # Logging, path incrementation, and CLI styling
├── train.py                     # Training loop with live diagnostics & checkpointing
├── test.py                      # Standalone evaluation script on validation/test sets
├── predict.py                   # Multi-modal inference & visualization overlay generator
├── evolve.py                    # Genetic hyperparameter evolution engine
├── run_experiments.ps1          # Automated sequential PowerShell benchmark runner
├── smoke_test.py                # Automated 15-stage comprehensive verification suite
├── environment.yml              # Conda environment definition for RTX 2060 Super (CUDA 12.1)
├── requirements.txt             # Pip environment dependencies
├── CONFIGS.md                   # Complete configuration reference & CLI overriding guide
└── CONTEXT.md                   # Domain vocabulary and glossary
```

---

## 🛰️ Multi-Modal Input Options

The framework supports dynamic concatenation (`concat`) of optical RGB imagery and continuous topographic rasters, as well as normalized element-wise addition (`add`) for scalar topography:

| Preset Name | Modalities Included | Channels | Fusion Mode | Description |
| :--- | :--- | :---: | :---: | :--- |
| `rgb_only` | `['IMAGE']` | 3 | Concat / Direct | Standard optical RGB imagery |
| `topo_only` | `['DTM_NORM', 'SLOPE', 'ASPECT']` | 4 | Concat | Topography only (1 DTM + 1 Slope + 2 Aspect sin/cos) |
| `rgb_dtm` | `['IMAGE', 'DTM_NORM']` | 4 / 3 | Concat / Add | RGB + Normalized DTM elevation |
| `rgb_slope` | `['IMAGE', 'SLOPE']` | 4 / 3 | Concat / Add | RGB + Slope steepness |
| `rgb_aspect` | `['IMAGE', 'ASPECT']` | 5 | Concat | RGB + Aspect orientation ($\sin\theta, \cos\theta$) |
| `all` | `['IMAGE', 'DTM_NORM', 'SLOPE', 'ASPECT']` | 7 | Concat | Full multi-modal feature set |
| `rgb_add_dtm` | `['IMAGE', 'DTM_NORM']` | 3 | Add | $(\text{RGB} + \text{DTM})/2 \in [0, 1]$ addition |
| `rgb_add_slope` | `['IMAGE', 'SLOPE']` | 3 | Add | $(\text{RGB} + \text{SLOPE})/2 \in [0, 1]$ addition |

Custom channel combinations can be passed directly via `--inputs`, e.g., `--inputs IMAGE DTM_NORM SLOPE`.

---

## ⚙️ Data Preprocessing & Topographic Normalization

1. **Aspect NoData / Flat Zero-Vector Encoding**:
   - In GIS DEM processing, undefined/flat terrain or NoData pixels (NaN, Inf, $-9999$, $-1$) are strictly encoded as **zero-magnitude vectors** $(0.0, 0.0)$, rather than mapping $\theta=0^\circ$ to True North $(0.0, 1.0)$.
   - Prevents artificial directional bias across flat plains and unmeasured boundaries.
2. **Aspect Directional Consistency Under Geometric Augmentations**:
   - Aspect is represented as continuous directional vector components $(\sin\theta, \cos\theta)$ where $\theta$ is azimuth clockwise from North.
   - Under geometric augmentations (`fliplr`, `flipud`, `rot90`), directional components undergo exact coordinate transformations:
     - **Horizontal Flip (`fliplr`, $x \to -x$)**: East-West component inverted ($\sin\theta \to -\sin\theta$), North-South ($\cos\theta$) preserved.
     - **Vertical Flip (`flipud`, $y \to -y$)**: North-South component inverted ($\cos\theta \to -\cos\theta$), East-West ($\sin\theta$) preserved.
     - **90° Rotations (`rot90`, $k \in \{1, 2, 3\}$)**: Continuous azimuth vector rotated by $-90^\circ, -180^\circ, +90^\circ$.
3. **Active `img_size` Spatial Dimension Enforcement**:
   - Continuous multi-channel feature tensors are resized to exact target dimensions $(C_\text{in}, \text{img\_size}, \text{img\_size})$ using **bilinear interpolation**.
   - Binary segmentation masks are resized using **nearest-neighbor interpolation**, guaranteeing discrete $\{0.0, 1.0\}$ ground-truth labels.
4. **Optical RGB (`IMAGE`)**: Scaled to $[0.0, 1.0]$ via $x / 255.0$.
5. **Normalized DTM (`DTM_NORM`)**: 16-bit unsigned integer scaled to $[0.0, 1.0]$ via $x / 65535.0$.
6. **Slope (`SLOPE`)**: Degrees $[0, 90]^\circ$ scaled to $[0.0, 1.0]$ via $\text{slope} / 90.0$.
7. **Ground Truth (`LABEL`)**: Binary mask mapped to $\{0.0, 1.0\}$.

---

## 🗺️ Geospatial Leakage Prevention & Regional Corridor Partitioning

In geospatial semantic segmentation, random patch splitting causes severe **spatial data leakage** because adjacent spatial tiles share nearly identical lithology, soil moisture, and slope.

- **Regional Corridor Splitting**: Entire linear survey sections (e.g. corridor `Chainage_17` for validation; `Chainages_1, 2, 18, 19` for training) are isolated to ensure true out-of-region generalization.
- **Audit Tool ([`data/spatial.py`](file:///D:/landslide/data/spatial.py))**:
  ```bash
  python -m data.spatial --data data/landslide.yaml --audit
  ```
  Analyzes sample prefixes and geographic bounds, reporting shared regions and minimum tile distance violations across splits.

---

## 🔀 Multi-Modal Fusion Modes

The framework supports two distinct fusion mechanisms selectable via `--fusion`:
- **Concatenation (`--fusion concat`) [Recommended]**: Modalities are stacked along the channel axis (e.g., RGB (3ch) + DTM (1ch) = 4 input channels; RGB + Topo = 7 input channels). Preserves individual modality identity and cross-channel gradients.
- **Normalized Element-wise Addition (`--fusion add`)**: Single-channel scalar topographic rasters (e.g., normalized DTM $\in [0, 1]$) are added element-wise into the 3 optical RGB channels and renormalized to prevent distribution shift:
  $$\text{fused} = \frac{\text{RGB} + \sum \text{Aux}}{1 + N_{\text{aux}}} \in [0.0, 1.0]$$
  > [!NOTE]
  > Additive fusion is strictly restricted to single-channel scalar rasters (such as DTM or SLOPE). Multi-channel directional modalities (like 2-channel ASPECT) must use `concat`.

---

## 📉 Loss Function & Evaluation Metrics

The training objective combines pixel-level classification (Binary Cross Entropy with Logits) and region-level boundary overlap (Soft Dice Loss):

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCEWithLogits}} + \beta \cdot \mathcal{L}_{\text{Dice}}$$

Loss weights $\alpha, \beta$ and `pos_weight` are configurable in `data/hyp.scratch.yaml` and serialized inside checkpoint bundles.

### Metric Suite
- **Foreground IoU (`fg_iou`)**: Landslide Jaccard Index $\frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$.
- **Background IoU (`bg_iou`)**: Non-landslide Jaccard Index $\frac{\text{TN}}{\text{TN} + \text{FP} + \text{FN}}$.
- **True Two-Class Mean IoU (`miou`)**: Macro average $\frac{\text{fg\_iou} + \text{bg\_iou}}{2}$.
- **Global / Micro Dice (`fg_dice`)**: Dataset-level pixel aggregation $\frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$.
- **Per-Image / Macro Dice (`fg_dice_macro`)**: Arithmetic mean of per-image Dice scores (matches `BCEDiceLoss` aggregation).
- **Precision, Recall, & Pixel Accuracy**: Comprehensive binary classification diagnostics.

---

## 🚀 Quickstart Guide

### 1. Conda Environment Setup (RTX 2060 Super)

Create and activate the optimized Conda environment (Python 3.10, PyTorch 2.3+, CUDA 12.1):

```bash
conda env create -f environment.yml
conda activate landslide
```

Alternatively, install via pip:
```bash
pip install -r requirements.txt
```

### 2. Verify Setup with Smoke Test
Run the automated test suite to verify datasets, models, loss functions, training, and inference:
```bash
python smoke_test.py
```

### 3. Model Training (Maximized GPU Utilization)

You can launch training using a pre-configured YAML file (`--config`), eliminating the need to type arguments repeatedly:

```bash
# 1. Pure 3-Channel RGB Training (Direct U-Net Without Projection)
python train.py --config configs/train_rgb_noproj.yaml

# 2. RGB + DTM Additive Fusion Training (Direct U-Net Without Projection)
python train.py --config configs/train_rgb_add_dtm.yaml

# 3. Standard High-Throughput Concatenation Training (RGB + DTM, 4 Channels)
python train.py --config configs/train.yaml

# 4. Full 7-Channel Multi-Modal Training
python train.py --config configs/train_multimodal.yaml

# 5. Fine-Tuning from Official Carvana U-Net Weights
python train.py --config configs/train_finetune_carvana.yaml
```

> [!TIP]
> You can override any configuration parameter on the fly:
> ```bash
> python train.py --config configs/train_rgb_add_dtm.yaml --epochs 100 --batch-size 16 --name custom_run
> ```
> See [`CONFIGS.md`](file:///D:/landslide/CONFIGS.md) for full configuration details.

### 4. Transfer Learning & Official Pretrained Weights
Initialize training from official U-Net pretrained weights (`milesial/Pytorch-UNet` Carvana weights) with automatic download and layer remapping:
```bash
python train.py --cfg models/architectures/unet.yaml --inputs rgb_only --weights unet_carvana --epochs 50 --batch-size 8
```

Supported pretrained aliases and sources:
- `--weights unet_carvana` / `unet_carvana_scale1.0`: Official PyTorch U-Net (scale 1.0 Carvana weights - default)
- `--weights unet_carvana_scale0.5`: Official PyTorch U-Net (scale 0.5 Carvana weights)
- `--weights https://...`: Direct checkpoint download URL
- `--weights runs/train/exp/weights/best.pt`: Local checkpoint from a previous experiment

### 5. Resuming Interrupted Training
Resume an interrupted run seamlessly (restores model, optimizer, scheduler, and epoch counter):
```bash
# Automatically resume the most recent run in runs/train/
python train.py --resume

# Or specify an explicit checkpoint path:
python train.py --resume runs/train/exp/weights/last.pt
```

### 6. Automated Sequential Benchmark Suite (PowerShell)
Run a systematic benchmark across combinations of **RGB-Only** vs. **RGB-DTM**, **Pure Direct U-Net** vs. **U-Net Proj**, and **Concat** vs. **Addition** fusion:
```powershell
# Run all 6 benchmark combinations sequentially
.\run_experiments.ps1 -Epochs 50 -BatchSize 8

# Run only selected experiments (e.g. Experiments 1, 3, and 6)
.\run_experiments.ps1 -SelectIds 1,3,6 -Epochs 50

# Dry-run preview without executing
.\run_experiments.ps1 -DryRun
```
Outputs an interactive comparison leaderboard and exports `runs/train/experiments_summary.csv`.

### 7. Genetic Hyperparameter Evolution
Automatically search the continuous hyperparameter space (`lr0`, `lrf`, `momentum`, `weight_decay`, `bce_weight`, `dice_weight`, `pos_weight`, augmentations) using genetic mutation:
```bash
python evolve.py --config configs/evolve.yaml --generations 30 --epochs 15
```

### 8. Evaluation
Evaluate checkpoint on the validation set using a configuration file or CLI flags:
```bash
python test.py --config configs/test.yaml
```

### 9. Inference & Visual Overlays
Generate segmentation masks and visual comparison heatmaps:
```bash
python predict.py --config configs/predict.yaml
```

Outputs:
- Binary PNG masks saved to `runs/predict/exp/masks/`
- 5-panel visual overlay comparison figures saved to `runs/predict/exp/overlays/`

