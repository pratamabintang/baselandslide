import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union, Callable
from PIL import Image
import numpy as np
import torch

from data.base import BaseMultiModalDataset, register_dataset
from data.transforms import MultiModalTransform


# Default modality configurations
DEFAULT_MODALITY_SPECS = {
    'IMAGE': {'channels': 3, 'ext': '.png', 'norm': 'rgb'},
    'DTM_NORM': {'channels': 1, 'ext': '.tif', 'norm': 'uint16'},
    'DTM': {'channels': 1, 'ext': '.tif', 'norm': 'minmax', 'bounds': [-2.40497, 170.77879]},
    'SLOPE': {'channels': 1, 'ext': '.tif', 'norm': 'degrees_90'},
    'ASPECT': {'channels': 2, 'ext': '.tif', 'norm': 'sincos'},
    'LABEL': {'channels': 1, 'ext': '.png', 'norm': 'binary_mask'},
}


class FolderStructureAdapter(BaseMultiModalDataset):
    """
    Generic Multi-Modal Adapter for folder-organized datasets.
    
    Expects directory structure:
    root_dir/split/
    ├── MODALITY_1/ (e.g. IMAGE/sample_001.png)
    ├── MODALITY_2/ (e.g. DTM_NORM/sample_001.tif)
    └── LABEL/      (e.g. LABEL/sample_001.png)
    """

    def __init__(self, root_dir: Union[str, Path], split: str = 'train',
                 inputs: Union[str, List[str]] = 'rgb_only',
                 img_size: Tuple[int, int] = (512, 512),
                 augment: bool = False, hyp: Optional[Dict] = None,
                 presets: Optional[Dict] = None,
                 modality_specs: Optional[Dict[str, Dict]] = None,
                 label_folder: str = 'LABEL',
                 cache_ram: bool = False):
        super().__init__(root_dir, split, inputs, img_size, augment, hyp, presets)

        self.split_dir = self.root_dir / split if (self.root_dir / split).exists() else self.root_dir
        self.label_folder = label_folder
        self.cache_ram = cache_ram or (hyp.get('cache_ram', False) if hyp else False)
        self._cache = {}

        # Merge modality specs
        self.specs = DEFAULT_MODALITY_SPECS.copy()
        if modality_specs:
            for k, v in modality_specs.items():
                if k in self.specs:
                    self.specs[k].update(v)
                else:
                    self.specs[k] = v

        self._active_inputs = self._resolve_inputs(inputs, presets)
        self._in_channels = sum(self.specs.get(mod, {'channels': 1})['channels'] for mod in self._active_inputs)

        # Index valid samples
        self._samples = self._find_valid_samples()
        if len(self._samples) == 0:
            raise RuntimeError(f"No matching samples found in {self.split_dir} for modalities {self._active_inputs}")

        self.transform = MultiModalTransform(augment=self.augment, hyp=self.hyp, img_size=self.img_size)


    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def active_inputs(self) -> List[str]:
        return self._active_inputs

    @property
    def samples(self) -> List[str]:
        return self._samples

    def _resolve_inputs(self, inputs, presets=None):
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
            split_inputs = [x.strip().upper() for x in inputs.replace(',', ' ').split() if x.strip()]
            return split_inputs if split_inputs else ['IMAGE']

        if isinstance(inputs, (list, tuple)):
            resolved = []
            for item in inputs:
                if isinstance(item, str) and item in merged_presets:
                    resolved.extend(merged_presets[item])
                else:
                    resolved.append(str(item).upper())
            return list(dict.fromkeys(resolved))

        return ['IMAGE']

    def _find_valid_samples(self) -> List[str]:
        label_dir = self.split_dir / self.label_folder
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        label_files = glob.glob(str(label_dir / '*.*'))
        stems = [Path(f).stem for f in label_files]

        valid_stems = []
        for stem in sorted(stems):
            all_present = True
            for mod in self._active_inputs:
                spec = self.specs.get(mod, {'ext': '.tif'})
                ext = spec.get('ext', '.tif')
                mod_path = self.split_dir / mod / f"{stem}{ext}"
                if not mod_path.exists():
                    all_present = False
                    break
            if all_present:
                valid_stems.append(stem)

        return valid_stems

    def load_modality_array(self, mod: str, stem: str) -> np.ndarray:
        """
        Load, impute NaNs, and normalize a single modality into a numpy array of shape (H, W, C_mod).
        """
        spec = self.specs.get(mod, {'channels': 1, 'ext': '.tif', 'norm': 'scale', 'scale_factor': 1.0})
        ext = spec.get('ext', '.tif')
        norm_type = spec.get('norm', 'scale')
        mod_path = self.split_dir / mod / f"{stem}{ext}"

        if not mod_path.exists():
            raise FileNotFoundError(f"Missing file for {mod}: {mod_path}")

        if norm_type == 'rgb' or (mod == 'IMAGE' and norm_type != 'custom'):
            img = Image.open(mod_path).convert('RGB')
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / 255.0
            return arr

        elif norm_type == 'uint16' or mod == 'DTM_NORM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) / 65535.0
            return np.expand_dims(arr, axis=-1)

        elif norm_type == 'minmax' or mod == 'DTM':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            bounds = spec.get('bounds', [-2.40497, 170.77879])
            d_min, d_max = bounds[0], bounds[1]
            arr = np.clip((arr - d_min) / (d_max - d_min + 1e-7), 0.0, 1.0)
            return np.expand_dims(arr, axis=-1)

        elif norm_type == 'degrees_90' or mod == 'SLOPE':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = np.clip(arr / 90.0, 0.0, 1.0)
            return np.expand_dims(arr, axis=-1)

        elif norm_type == 'sincos' or mod == 'ASPECT':
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            rad = arr * (np.pi / 180.0)
            sin_aspect = np.sin(rad)
            cos_aspect = np.cos(rad)
            return np.stack([sin_aspect, cos_aspect], axis=-1)

        else:
            img = Image.open(mod_path)
            arr = np.array(img, dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            scale = spec.get('scale_factor', 1.0)
            arr = arr / scale
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
            return arr

    def load_label_mask(self, stem: str) -> np.ndarray:
        """Load ground truth mask and return float binary mask (H, W)."""
        label_ext = self.specs.get('LABEL', {}).get('ext', '.png')
        label_path = self.split_dir / self.label_folder / f"{stem}{label_ext}"
        img = Image.open(label_path)
        arr = np.array(img)
        mask = (arr > 0).astype(np.float32)
        return mask

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        stem = self._samples[idx]

        if self.cache_ram and stem in self._cache:
            multi_channel_tensor, label_mask = self._cache[stem]
            # Copy to avoid modifying cached arrays in-place during spatial augmentations
            multi_channel_tensor = multi_channel_tensor.copy()
            label_mask = label_mask.copy()
        else:
            # Load and stack modalities
            mod_arrays = [self.load_modality_array(mod, stem) for mod in self._active_inputs]
            multi_channel_tensor = np.concatenate(mod_arrays, axis=-1)
            label_mask = self.load_label_mask(stem)

            if self.cache_ram:
                self._cache[stem] = (multi_channel_tensor.copy(), label_mask.copy())

        # Apply synchronized spatial augmentations
        tensor, mask = self.transform(multi_channel_tensor, label_mask)

        return tensor, mask, stem



class CustomDatasetWrapper(BaseMultiModalDataset):
    """
    Adapter to wrap an arbitrary custom PyTorch Dataset or generator function
    into the standard BaseMultiModalDataset interface.
    """

    def __init__(self, raw_dataset: Any, in_channels: int, active_inputs: Optional[List[str]] = None):
        self.raw_dataset = raw_dataset
        self._in_channels = in_channels
        self._active_inputs = active_inputs or [f"CH_{i}" for i in range(in_channels)]

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def active_inputs(self) -> List[str]:
        return self._active_inputs

    @property
    def samples(self) -> List[str]:
        if hasattr(self.raw_dataset, 'samples'):
            return self.raw_dataset.samples
        return [f"sample_{i:06d}" for i in range(len(self.raw_dataset))]

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.raw_dataset[idx]

        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                tensor, mask, sample_id = item
            elif len(item) == 2:
                tensor, mask = item
                sample_id = f"sample_{idx:06d}"
            else:
                raise ValueError(f"Unexpected item tuple length: {len(item)}")
        elif isinstance(item, dict):
            tensor = item.get('image', item.get('tensor', item.get('input')))
            mask = item.get('mask', item.get('label', item.get('target')))
            sample_id = str(item.get('id', item.get('stem', f"sample_{idx:06d}")))
        else:
            raise TypeError(f"Cannot adapt dataset item of type {type(item)}")

        # Ensure tensor format
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.array(tensor)).float()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(np.array(mask)).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        self.validate_sample_output(tensor, mask, sample_id)
        return tensor, mask, str(sample_id)
