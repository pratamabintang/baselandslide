# Multi-Modal Landslide Segmentation Baseline

A modular, declarative PyTorch repository for multi-modal landslide segmentation using U-Net architectures, declarative YAML layer parsing, dynamic $1\times1$ input projection, and combined BCE + Dice loss.

---

## 📁 Repository Structure

```text
landslide/
├── dataset/
│   └── dataset_1/               # Primary multi-modal landslide dataset
│       ├── train/               # Training rasters (IMAGE, LABEL, DTM_NORM, SLOPE, ASPECT)
│       ├── validation/          # Validation rasters (IMAGE, LABEL, DTM_NORM, SLOPE, ASPECT)
│       ├── global_raster_minmax.json
│       └── pairing_reports/
├── data/
│   ├── landslide.yaml           # Dataset configuration, paths, classes, presets
│   ├── hyp.scratch.yaml         # Training hyperparameters & loss weights
│   ├── transforms.py            # Synchronized multi-modal spatial augmentations
│   └── datasets.py              # Multi-modal dataset loader & NaN/normalization engine
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
├── train.py                     # Training loop with validation & best model checkpointing
├── test.py                      # Standalone evaluation script on validation/test sets
├── predict.py                   # Multi-modal inference & visualization overlay generator
├── requirements.txt             # Environment dependencies
├── CONTEXT.md                   # Domain vocabulary and glossary
└── docs/adr/                    # Architecture Decision Records
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

You can also pass arbitrary custom combinations via `--inputs`, e.g., `--inputs IMAGE DTM_NORM SLOPE`.

---

## ⚙️ Data Preprocessing & Normalization

1. **NaN Handling**: NaN pixels at raster boundaries are replaced immediately upon loading via `np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)`.
2. **Optical RGB (`IMAGE`)**: Scaled to $[0, 1]$ via $x / 255.0$.
3. **Normalized DTM (`DTM_NORM`)**: 16-bit unsigned integer scaled to $[0, 1]$ via $x / 65535.0$.
4. **Slope (`SLOPE`)**: Degrees $[0, 90]^\circ$ scaled to $[0, 1]$ via $\text{slope} / 90.0$.
5. **Aspect (`ASPECT`)**: Degrees $[0, 360]^\circ$ decomposed into cyclic continuous components $\sin(\theta)$ and $\cos(\theta)$ in $[-1, 1]$ (2 channels).
6. **Ground Truth (`LABEL`)**: Binary mask mapped to $\{0.0, 1.0\}$.

---

## 🧱 Architecture & Model Parser

Neural architectures are declared in YAML (`models/unet.yaml`) using `[from, number, module, args]` blocks following the SSFusion paradigm. 

The first layer (Layer 0) uses a standard $1\times1$ `Conv` block to project arbitrary concatenated multi-modal tensors ($C_{\text{in}} \in [1, 7]$) down to 3 channels before the U-Net encoder:

```yaml
# models/unet.yaml
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

## 📉 Loss Function

The training objective combines pixel-level classification (Binary Cross Entropy with Logits) and region-level boundary overlap (Soft Dice Loss):

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCEWithLogits}} + \beta \cdot \mathcal{L}_{\text{Dice}}$$

Loss weights $\alpha, \beta$ are configurable in `data/hyp.scratch.yaml`.

---

## 🚀 Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training
Train a model with RGB + DTM inputs:
```bash
python train.py --cfg models/architectures/unet.yaml --inputs rgb_dtm --epochs 50 --batch-size 8 --device 0
```

Train with Topography only on a lighter architecture:
```bash
python train.py --cfg models/architectures/unet_lite.yaml --inputs topo_only --epochs 50 --batch-size 8 --device 0
```

### 3. Evaluation
Evaluate checkpoint on the validation set:
```bash
python test.py --weights runs/train/exp/weights/best.pt --data data/landslide.yaml --split val --conf-thres 0.5
```

### 4. Inference & Visual Overlays
Generate segmentation masks and visual comparison heatmaps:
```bash
python predict.py --weights runs/train/exp/weights/best.pt --source dataset/dataset_1/validation --conf-thres 0.5
```
Outputs:
- Binary PNG masks saved to `runs/predict/exp/masks/`
- 3-panel visual overlay figures saved to `runs/predict/exp/overlays/`
