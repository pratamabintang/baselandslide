# Multi-Modal Landslide Segmentation Baseline

A modular, declarative PyTorch repository for multi-modal landslide segmentation using U-Net architectures, declarative YAML layer parsing, dynamic $1\times1$ input projection, and combined BCE + Dice loss.

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
│   ├── transforms.py            # Synchronized multi-modal spatial augmentations
│   └── datasets.py              # Multi-modal dataset factory & loader
├── models/
│   ├── architectures/
│   │   ├── unet.yaml            # Standard U-Net architecture YAML
│   │   └── unet_lite.yaml       # Lightweight U-Net architecture YAML
│   ├── common.py                # Layer modules: Conv, DoubleConv, Down, Up, OutConv, Concat
│   └── model_parser.py          # Declarative YAML architecture parser & Model class
├── utils/
│   ├── loss.py                  # Combined BCEWithLogitsLoss + Soft Dice Loss
│   ├── metrics.py               # Segmentation metrics (mIoU, Dice/F1, Precision, Recall, Accuracy)
│   ├── plots.py                 # Loss curves and multi-panel prediction overlays
│   ├── torch_utils.py           # Device management, seeding, and early stopping
│   └── general.py               # Logging, path incrementation, and CLI styling
├── train.py                     # Training loop with live diagnostics & checkpointing
├── test.py                      # Standalone evaluation script on validation/test sets
├── predict.py                   # Multi-modal inference & visualization overlay generator
├── smoke_test.py                # Automated 6-stage repository verification suite
├── environment.yml              # Conda environment definition for RTX 2060 Super (CUDA 12.1)
├── requirements.txt             # Pip environment dependencies
└── CONTEXT.md                   # Domain vocabulary and glossary
```

---

## 🛰️ Multi-Modal Input Options

The framework supports dynamic concatenation of optical RGB imagery and continuous topographic rasters:

| Preset Name | Modalities Included | Channels | Description |
| :--- | :--- | :---: | :--- |
| `rgb_only` | `['IMAGE']` | 3 | Standard optical RGB imagery |
| `topo_only` | `['DTM_NORM', 'SLOPE', 'ASPECT']` | 4 | Topography only (1 DTM + 1 Slope + 2 Aspect sin/cos) |
| `rgb_dtm` | `['IMAGE', 'DTM_NORM']` | 4 | RGB + Normalized DTM elevation |
| `rgb_slope` | `['IMAGE', 'SLOPE']` | 4 | RGB + Slope steepness |
| `rgb_aspect` | `['IMAGE', 'ASPECT']` | 5 | RGB + Aspect orientation ($\sin\theta, \cos\theta$) |
| `all` | `['IMAGE', 'DTM_NORM', 'SLOPE', 'ASPECT']` | 7 | Full multi-modal feature set |

Custom channel combinations can be passed directly via `--inputs`, e.g., `--inputs IMAGE DTM_NORM SLOPE`.

---

## ⚙️ Data Preprocessing & Normalization

1. **NaN Handling**: Edge NaNs are replaced immediately upon loading via `np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`.
2. **Optical RGB (`IMAGE`)**: Scaled to $[0.0, 1.0]$ via $x / 255.0$.
3. **Normalized DTM (`DTM_NORM`)**: 16-bit unsigned integer scaled to $[0.0, 1.0]$ via $x / 65535.0$.
4. **Slope (`SLOPE`)**: Degrees $[0, 90]^\circ$ scaled to $[0.0, 1.0]$ via $\text{slope} / 90.0$.
5. **Aspect (`ASPECT`)**: Degrees $[0, 360]^\circ$ decomposed into continuous components $\sin(\theta)$ and $\cos(\theta)$ in $[-1.0, 1.0]$ (2 channels).
6. **Ground Truth (`LABEL`)**: Binary mask mapped to $\{0.0, 1.0\}$.

---

## 🔌 Integrating New External Datasets

Any new dataset can be integrated by adhering to the `BaseMultiModalDataset` interface (`data/base.py`):

### 1. Expected Interface Contract
Every sample returned by `__getitem__(idx)` must be a 3-element tuple:
$$\text{sample} = (\text{tensor}, \text{mask}, \text{sample\_id})$$
- `tensor`: `torch.FloatTensor` of shape `(C_in, H, W)` with normalized feature values.
- `mask`: `torch.FloatTensor` of shape `(1, H, W)` with binary ground truth $\{0.0, 1.0\}$.
- `sample_id`: `str` unique identifier.

### 2. Integration Methods
- **Zero-Code Folder Datasets**: Create a dataset YAML referencing your new folder (e.g. `data/my_dataset.yaml` with `dataset_type: folder`).
- **Custom PyTorch Classes**: Register custom classes with `@register_dataset("my_custom_name")` in `data/adapters.py`.

---

## 🧱 Architecture & Model Parser

Neural architectures are declared in YAML (`models/architectures/unet.yaml`) using `[from, number, module, args]` blocks following the SSFusion paradigm. 

The first layer (Layer 0) uses a standard $1\times1$ `Conv` block to project arbitrary concatenated multi-modal tensors ($C_{\text{in}} \in [1, 7]$) down to 3 channels before the U-Net encoder:

```yaml
# models/architectures/unet.yaml
nc: 1
backbone:
  [[-1, 1, Conv, [3, 1, 1]],         # 0 - (C_in -> 3 via 1x1 Conv)
   [-1, 1, DoubleConv, [64]],        # 1 - Encoder Stage 1 (3 -> 64)
   [-1, 1, Down, [128]],             # 2 - Encoder Stage 2 (64 -> 128)
   [-1, 1, Down, [256]],             # 3 - Encoder Stage 3 (128 -> 256)
   [-1, 1, Down, [512]],             # 4 - Encoder Stage 4 (256 -> 512)
   [-1, 1, Down, [1024]],            # 5 - Bottleneck (512 -> 1024)
  ]
head:
  [[[5, 4], 1, Up, [512]],           # 6 - Decoder Stage 4 (Up 5 + Skip 4 -> 512)
   [[6, 3], 1, Up, [256]],           # 7 - Decoder Stage 3 (Up 6 + Skip 3 -> 256)
   [[7, 2], 1, Up, [128]],           # 8 - Decoder Stage 2 (Up 7 + Skip 2 -> 128)
   [[8, 1], 1, Up, [64]],            # 9 - Decoder Stage 1 (Up 8 + Skip 1 -> 64)
   [-1, 1, OutConv, [1]],            # 10 - Output Logits (64 -> nc)
  ]
```

---

## 📉 Loss Function & Evaluation Metrics

The training objective combines pixel-level classification (Binary Cross Entropy with Logits) and region-level boundary overlap (Soft Dice Loss):

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCEWithLogits}} + \beta \cdot \mathcal{L}_{\text{Dice}}$$

Loss weights $\alpha, \beta$ are configurable in `data/hyp.scratch.yaml`.

Evaluation tracks:
- **Mean IoU (Jaccard Index)**
- **Dice Coefficient (F1-Score)**
- **Precision & Recall**
- **Pixel Accuracy**

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
Run the automated 6-stage test suite to verify datasets, models, loss functions, training, and inference:
```bash
python smoke_test.py
```

### 3. Model Training (Maximized GPU Utilization)

You can launch training using a pre-configured YAML file (`--config`), eliminating the need to type arguments repeatedly:

```bash
# Standard High-Throughput Training (RGB + DTM)
python train.py --config configs/train.yaml

# Full 7-Channel Multi-Modal Training
python train.py --config configs/train_multimodal.yaml

# Fine-Tuning from Official Carvana U-Net Weights
python train.py --config configs/train_finetune_carvana.yaml
```

> [!TIP]
> You can override any configuration parameter on the fly:
> ```bash
> python train.py --config configs/train.yaml --epochs 100 --batch-size 16 --name custom_run
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

### 6. Evaluation
Evaluate checkpoint on the validation set using a configuration file or CLI flags:
```bash
python test.py --config configs/test.yaml
```

### 7. Inference & Visual Overlays
Generate segmentation masks and visual comparison heatmaps:
```bash
python predict.py --config configs/predict.yaml
```


Outputs:
- Binary PNG masks saved to `runs/predict/exp/masks/`
- 3-panel visual overlay figures saved to `runs/predict/exp/overlays/`

