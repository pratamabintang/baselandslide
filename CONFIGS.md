# ⚙️ Configuration Files Reference (`--config`)

All primary scripts (`train.py`, `test.py`, `predict.py`) support loading arguments directly from YAML configuration files using the `--config` flag.

---

## 🚀 Quick Reference

| Command | File | Description |
| :--- | :--- | :--- |
| `python train.py --config configs/train_rgb_noproj.yaml` | [`configs/train_rgb_noproj.yaml`](file:///D:/landslide/configs/train_rgb_noproj.yaml) | Pure 3-channel RGB baseline without projection layer |
| `python train.py --config configs/train_rgb_add_dtm.yaml` | [`configs/train_rgb_add_dtm.yaml`](file:///D:/landslide/configs/train_rgb_add_dtm.yaml) | RGB + DTM element-wise addition fusion (3 channels) |
| `python train.py --config configs/train.yaml` | [`configs/train.yaml`](file:///D:/landslide/configs/train.yaml) | Standard training (RGB + DTM concatenation) on RTX 2060 Super |
| `python train.py --config configs/train_multimodal.yaml` | [`configs/train_multimodal.yaml`](file:///D:/landslide/configs/train_multimodal.yaml) | Full 7-channel multi-modal training |
| `python train.py --config configs/train_finetune_carvana.yaml` | [`configs/train_finetune_carvana.yaml`](file:///D:/landslide/configs/train_finetune_carvana.yaml) | Fine-tuning from official Carvana U-Net weights |
| `python test.py --config configs/test.yaml` | [`configs/test.yaml`](file:///D:/landslide/configs/test.yaml) | Model evaluation on validation split |
| `python predict.py --config configs/predict.yaml` | [`configs/predict.yaml`](file:///D:/landslide/configs/predict.yaml) | Inference and visualization overlay generation |
| `python evolve.py --config configs/evolve.yaml` | [`configs/evolve.yaml`](file:///D:/landslide/configs/evolve.yaml) | Genetic hyperparameter evolution & auto-tuning |

---

## ⚡ How It Works

1. **Load Defaults from File**: When you specify `--config path/to/config.yaml`, all training, evaluation, or inference settings are pre-loaded from the YAML file.
2. **Dynamic CLI Overrides**: Any argument explicitly passed on the command line will override the value in the YAML configuration on the fly.

### Example:
```bash
# Run pure 3-channel RGB without projection
python train.py --config configs/train_rgb_noproj.yaml

# Run RGB + DTM addition fusion without projection
python train.py --config configs/train_rgb_add_dtm.yaml

# Override epochs and batch size on the fly
python train.py --config configs/train_rgb_add_dtm.yaml --epochs 100 --batch-size 16 --name custom_exp
```

---

## 📁 Pre-Built Configuration Files

### 1. [`configs/train_rgb_noproj.yaml`](file:///D:/landslide/configs/train_rgb_noproj.yaml) (Pure 3-Channel RGB Without Projection)
```yaml
cfg: models/architectures/unet_noproj.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml
inputs: rgb_only                  # Pure 3-channel RGB (IMAGE)
weights: ''                       # Pretrained weights path or alias (e.g. unet_carvana)
resume: false

epochs: 50
batch_size: 12
img_size: 512
conf_thres: 0.5

device: '0'
workers: 2
cache_ram: true
accumulate: 1
deterministic: false

project: runs/train
name: exp_rgb_noproj
exist_ok: false
patience: 15
seed: 42
```

---

### 2. [`configs/train_rgb_add_dtm.yaml`](file:///D:/landslide/configs/train_rgb_add_dtm.yaml) (RGB + DTM Additive Fusion Without Projection)
```yaml
cfg: models/architectures/unet_noproj.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml
inputs: rgb_dtm                   # IMAGE (3ch) + DTM_NORM (1ch)
fusion: add                       # Element-wise addition: RGB [0,1] + DTM [0,1] -> 3 channels
weights: ''
resume: false

epochs: 50
batch_size: 12
img_size: 512
conf_thres: 0.5

device: '0'
workers: 2
cache_ram: true
accumulate: 1
deterministic: false

project: runs/train
name: exp_rgb_add_dtm
exist_ok: false
patience: 15
seed: 42
```

---

### 3. [`configs/train.yaml`](file:///D:/landslide/configs/train.yaml) (Standard Concatenation Training)
```yaml
cfg: models/architectures/unet.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml
inputs: rgb_dtm                   # Presets: rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all
fusion: concat                    # Concatenation mode -> 4 channels input
weights: ''                       # Pretrained weights path or alias (e.g. unet_carvana)
resume: false                     # Set true or checkpoint path to resume

epochs: 50
batch_size: 12                    # Optimal for RTX 2060 Super (8GB VRAM)
img_size: 512
conf_thres: 0.5

device: '0'
workers: 2
cache_ram: true                   # RAM caching for zero disk I/O latency
accumulate: 1
deterministic: false              # Enables cuDNN benchmark auto-tuner

project: runs/train
name: exp_rgb_dtm
exist_ok: false
patience: 15
seed: 42
```

---

### 4. [`configs/train_multimodal.yaml`](file:///D:/landslide/configs/train_multimodal.yaml) (Full 7-Channel)
```yaml
cfg: models/architectures/unet.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml
inputs: all                       # IMAGE(3) + DTM_NORM(1) + SLOPE(1) + ASPECT(2)
weights: ''
resume: false

epochs: 50
batch_size: 12
img_size: 512
conf_thres: 0.5

device: '0'
workers: 2
cache_ram: true
accumulate: 1
deterministic: false

project: runs/train
name: exp_multimodal_all
exist_ok: false
patience: 15
seed: 42
```

---

### 5. [`configs/test.yaml`](file:///D:/landslide/configs/test.yaml) (Evaluation)
```yaml
weights: runs/train/exp_rgb_dtm/weights/best.pt
cfg: models/architectures/unet.yaml
data: data/landslide.yaml
inputs: rgb_dtm
split: val
batch_size: 12
img_size: 512
conf_thres: 0.5
device: '0'
workers: 2
plots: true

project: runs/val
name: exp_val
exist_ok: false
```

---

### 6. [`configs/predict.yaml`](file:///D:/landslide/configs/predict.yaml) (Inference)
```yaml
weights: runs/train/exp_rgb_dtm/weights/best.pt
source: dataset/dataset_1/validation
cfg: models/architectures/unet.yaml
data: data/landslide.yaml
inputs: rgb_dtm
conf_thres: 0.5
device: '0'

project: runs/predict
name: exp_pred
exist_ok: false
```

---

### 7. [`configs/evolve.yaml`](file:///D:/landslide/configs/evolve.yaml) (Hyperparameter Evolution)
```yaml
generations: 30                    # Total candidate generations to evolve
epochs: 15                         # Fast screening budget per candidate
mutation_prob: 0.8                 # Probability of mutating each individual gene

cfg: models/architectures/unet.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml         # Baseline seed hyperparameters
inputs: rgb_dtm
weights: unet_carvana
fusion: concat

batch_size: 8
img_size: 512
device: '0'
workers: 2
cache_ram: false

project: runs/evolve
name: exp_evolve
exist_ok: false
resume: false
```
