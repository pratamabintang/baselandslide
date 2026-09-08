import os
import glob
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from data.transforms import MultiModalTransform


MODALITY_CHANNELS = {
    'IMAGE': 3,
    'DTM': 1,
    'DTM_NORM': 1,
    'SLOPE': 1,
    'ASPECT': 2,
}

MODALITY_EXTENSIONS = {
    'IMAGE': '.png',
    'DTM': '.tif',
    'DTM_NORM': '.tif',
    'SLOPE': '.tif',
    'ASPECT': '.tif',
    'LABEL': '.png',
}


def resolve_inputs(inputs, presets=None):
    """
    Resolve input preset name or list of modal names into a validated canonical list.

    Args:
        inputs (str or list): e.g. 'rgb_only', 'topo_only', or ['IMAGE', 'DTM_NORM']
        presets (dict, optional): preset dictionary from data.yaml

    Returns:
        list of str: e.g. ['IMAGE', 'DTM_NORM']
    """
    default_presets = {
        'rgb_only': ['IMAGE'],
        'topo_only': ['DTM_NORM', 'SLOPE', 'ASPECT'],
        'rgb_dtm': ['IMAGE', 'DTM_NORM'],
        'rgb_slope': ['IMAGE', 'SLOPE'],
        'rgb_aspect': ['IMAGE', 'ASPECT'],
        'all': ['IMAGE', 'DTM_NORM', 'SLOPE', 'ASPECT'],
    }
    merged_presets = default_presets.copy()
    if presets:
        merged_presets.update(presets)

    if isinstance(inputs, str):
        if inputs in merged_presets:
            return merged_presets[inputs]
        # Allow comma/space-separated string e.g. 'IMAGE,DTM_NORM'
        split_inputs = [x.strip().upper() for x in inputs.replace(',', ' ').split() if x.strip()]
        if split_inputs:
            return split_inputs
        return ['IMAGE']

    if isinstance(inputs, (list, tuple)):
        resolved = []
        for item in inputs:
            if isinstance(item, str) and item in merged_presets:
                resolved.extend(merged_presets[item])
            else:
                resolved.append(str(item).upper())
        return list(dict.fromkeys(resolved))  # preserve order, remove duplicates

    return ['IMAGE']


def get_channel_count(inputs, presets=None):
    """Calculate total input channels for the resolved input modalities."""
    resolved = resolve_inputs(inputs, presets)
    total_ch = sum(MODALITY_CHANNELS.get(mod, 1) for mod in resolved)
    return total_ch


class LandslideDataset(Dataset):
    """
    Multi-Modal Landslide Segmentation Dataset.
    Loads RGB optical imagery and terrain rasters (DTM_NORM, SLOPE, ASPECT), applies
    NaN edge imputation, normalized representations (e.g. sin/cos aspect),
    and concatenates them along the channel dimension.
    """

    def __init__(self, root_dir, split='train', inputs='rgb_only', img_size=(512, 512),
                 augment=False, hyp=None, presets=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split if (self.root_dir / split).exists() else self.root_dir
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        self.hyp = hyp or {}
        self.presets = presets

        self.inputs = resolve_inputs(inputs, presets)
        self.in_channels = get_channel_count(self.inputs, presets)

        # Index dataset samples by stem matching across available folders
        self.samples = self._find_valid_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid paired samples found in {self.split_dir} for inputs: {self.inputs}")

        self.transform = MultiModalTransform(augment=self.augment, hyp=self.hyp, img_size=self.img_size)

    def _find_valid_samples(self):
        """Find matching sample IDs across active modalities and label directory."""
        label_dir = self.split_dir / 'LABEL'
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found at {label_dir}")

        label_files = glob.glob(str(label_dir / '*.*'))
        sample_stems = [Path(f).stem for f in label_files]

        # Verify existence in each active modality
        valid_stems = []
        for stem in sorted(sample_stems):
            all_present = True
            for mod in self.inputs:
                mod_dir = self.split_dir / mod
                ext = MODALITY_EXTENSIONS.get(mod, '.tif')
                mod_path = mod_dir / f"{stem}{ext}"
                if not mod_path.exists():
                    all_present = False
                    break
            if all_present:
                valid_stems.append(stem)

        return valid_stems

    def _load_modality(self, mod, stem):
        """Load, impute NaNs, and normalize a specific modality raster."""
        mod_dir = self.split_dir / mod
        ext = MODALITY_EXTENSIONS.get(mod, '.tif')
        mod_path = mod_dir / f"{stem}{ext}"

        if not mod_path.exists():
            raise FileNotFoundError(f"Missing file for modality {mod}: {mod_path}")

        if mod == 'IMAGE':
            img = Image.open(mod_path).convert('RGB')
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = arr / 255.0  # (H, W, 3) in [0, 1]
            return arr

        elif mod == 'DTM_NORM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = arr / 65535.0  # (H, W) in [0, 1]
            return np.expand_dims(arr, axis=-1)

        elif mod == 'DTM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            # Min-Max scale based on global dataset bounds [-2.405, 170.779]
            d_min, d_max = -2.40497088432312, 170.77879333496094
            arr = np.clip((arr - d_min) / (d_max - d_min + 1e-7), 0.0, 1.0)
            return np.expand_dims(arr, axis=-1)

        elif mod == 'SLOPE':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr / 90.0, 0.0, 1.0)  # (H, W) in [0, 1]
            return np.expand_dims(arr, axis=-1)

        elif mod == 'ASPECT':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            rad = arr * (np.pi / 180.0)
            sin_aspect = np.sin(rad)  # range [-1, 1]
            cos_aspect = np.cos(rad)  # range [-1, 1]
            stacked = np.stack([sin_aspect, cos_aspect], axis=-1)  # (H, W, 2)
            return stacked

        else:
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            return arr

    def _load_label(self, stem):
        """Load ground truth segmentation mask and map to binary float {0.0, 1.0}."""
        label_dir = self.split_dir / 'LABEL'
        label_path = label_dir / f"{stem}{MODALITY_EXTENSIONS['LABEL']}"
        img = Image.open(label_path)
        arr = np.array(img)
        # Value 65535 or > 0 is landslide (1.0), 0 is background (0.0)
        mask = (arr > 0).astype(np.float32)
        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem = self.samples[idx]

        # Load and concatenate active input modalities
        mod_arrays = [self._load_modality(mod, stem) for mod in self.inputs]
        multi_channel_tensor = np.concatenate(mod_arrays, axis=-1)  # (H, W, C_total)

        # Load label
        label_mask = self._load_label(stem)  # (H, W)

        # Apply synchronized transform
        tensor, mask = self.transform(multi_channel_tensor, label_mask)

        return tensor, mask, stem


def create_dataloader(data_yaml_path, split='train', inputs='rgb_only', batch_size=8,
                      img_size=512, augment=False, hyp=None, shuffle=True, num_workers=2,
                      pin_memory=True):
    """
    Construct DataLoader for LandslideDataset given a dataset YAML configuration.
    """
    if isinstance(data_yaml_path, (str, Path)):
        with open(data_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
    else:
        data_dict = data_yaml_path

    root_path = Path(data_dict.get('path', '.'))
    split_folder = data_dict.get(split, split)
    split_dir = root_path / split_folder if not Path(split_folder).is_absolute() else Path(split_folder)
    presets = data_dict.get('presets', {})

    dataset = LandslideDataset(
        root_dir=split_dir,
        split=split,
        inputs=inputs,
        img_size=img_size,
        augment=augment,
        hyp=hyp,
        presets=presets
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == 'train' and len(dataset) > batch_size)
    )

    return dataloader, dataset
