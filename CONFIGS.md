# ⚙️ Configuration Files Reference (`--config`)

All primary scripts (`train.py`, `test.py`, `predict.py`) support loading arguments directly from YAML configuration files using the `--config` flag.

---

## 🚀 Quick Reference

| Command | File | Description |
| :--- | :--- | :--- |
| `python train.py --config configs/train.yaml` | [`configs/train.yaml`](file:///D:/landslide/configs/train.yaml) | High-throughput training (RGB + DTM) on RTX 2060 Super |
| `python train.py --config configs/train_multimodal.yaml` | [`configs/train_multimodal.yaml`](file:///D:/landslide/configs/train_multimodal.yaml) | Full 7-channel multi-modal training |
| `python train.py --config configs/train_finetune_carvana.yaml` | [`configs/train_finetune_carvana.yaml`](file:///D:/landslide/configs/train_finetune_carvana.yaml) | Fine-tuning from official Carvana U-Net weights |
| `python test.py --config configs/test.yaml` | [`configs/test.yaml`](file:///D:/landslide/configs/test.yaml) | Model evaluation on validation split |
| `python predict.py --config configs/predict.yaml` | [`configs/predict.yaml`](file:///D:/landslide/configs/predict.yaml) | Inference and visualization overlay generation |

---

## ⚡ How It Works

1. **Load Defaults from File**: When you specify `--config path/to/config.yaml`, all training, evaluation, or inference settings are pre-loaded from the YAML file.
2. **Dynamic CLI Overrides**: Any argument explicitly passed on the command line will override the value in the YAML configuration on the fly.

### Example:
```bash
# Run with config defaults (50 epochs, batch size 12, rgb_dtm inputs)
python train.py --config configs/train.yaml

# Override epochs and batch size without editing the YAML file
python train.py --config configs/train.yaml --epochs 100 --batch-size 16 --name custom_exp
```

---

## 📁 Pre-Built Configuration Files

### 1. [`configs/train.yaml`](file:///D:/landslide/configs/train.yaml) (Standard Training)
```yaml
cfg: models/architectures/unet.yaml
data: data/landslide.yaml
hyp: data/hyp.scratch.yaml
inputs: rgb_dtm                   # Presets: rgb_only, topo_only, rgb_dtm, rgb_slope, rgb_aspect, all
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

### 2. [`configs/train_multimodal.yaml`](file:///D:/landslide/configs/train_multimodal.yaml) (Full 7-Channel)
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

### 3. [`configs/test.yaml`](file:///D:/landslide/configs/test.yaml) (Evaluation)
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

### 4. [`configs/predict.yaml`](file:///D:/landslide/configs/predict.yaml) (Inference)
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
