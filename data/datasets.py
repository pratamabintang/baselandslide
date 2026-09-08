import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import yaml
import torch
from torch.utils.data import DataLoader

from data.base import BaseMultiModalDataset, register_dataset, DATASET_REGISTRY
from data.adapters import FolderStructureAdapter, CustomDatasetWrapper, DEFAULT_MODALITY_SPECS


# Mapping of default modality channels
MODALITY_CHANNELS = {k: v['channels'] for k, v in DEFAULT_MODALITY_SPECS.items()}
MODALITY_EXTENSIONS = {k: v['ext'] for k, v in DEFAULT_MODALITY_SPECS.items()}


def resolve_inputs(inputs, presets=None):
    """
    Resolve input preset name or list of modal names into a validated canonical list.
    """
    default_presets = {
        'rgb_only': ['IMAGE'],
        'topo_only': ['DTM_NORM', 'SLOPE', 'ASPECT'],
        'rgb_dtm': ['IMAGE', 'DTM_NORM'],
        'rgb_slope': ['IMAGE', 'SLOPE'],
        'rgb_aspect': ['IMAGE', 'ASPECT'],
        'all': ['IMAGE', 'DTM_NORM', 'SLOPE', 'ASPECT'],
        'rgb_add_dtm': ['IMAGE', 'DTM_NORM'],
        'rgb_add_slope': ['IMAGE', 'SLOPE'],
        'rgb_add_aspect': ['IMAGE', 'ASPECT'],
        'rgb+dtm': ['IMAGE', 'DTM_NORM'],
        'rgb+slope': ['IMAGE', 'SLOPE'],
        'rgb+aspect': ['IMAGE', 'ASPECT'],
    }
    merged_presets = default_presets.copy()
    if presets:
        merged_presets.update(presets)

    if isinstance(inputs, str):
        if inputs in merged_presets:
            return merged_presets[inputs]
        split_inputs = [x.strip().upper() for x in inputs.replace(',', ' ').replace('+', ' ').split() if x.strip()]
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


def get_channel_count(inputs, presets=None, modality_specs=None, fusion='concat'):
    """
    Calculate total input channels for the resolved input modalities.
    When fusion is 'add', auxiliary channels are added element-wise into the 3-channel RGB base (output = 3 channels).
    """
    if str(fusion).lower() in ['add', 'addition', 'sum']:
        return 3

    resolved = resolve_inputs(inputs, presets)
    specs = DEFAULT_MODALITY_SPECS.copy()
    if modality_specs:
        specs.update(modality_specs)

    total_ch = sum(specs.get(mod, {'channels': 1})['channels'] for mod in resolved)
    return total_ch


@register_dataset('landslide')
@register_dataset('folder')
class LandslideDataset(FolderStructureAdapter):
    """
    Primary Landslide Segmentation Dataset Adapter.
    Inherits from FolderStructureAdapter to provide normalized multi-modal tensor outputs.
    """
    pass


def build_dataset(data_yaml_path: Union[str, Path, Dict],
                  split: str = 'train',
                  inputs: Union[str, List[str]] = 'rgb_only',
                  img_size: Union[int, Tuple[int, int]] = (512, 512),
                  augment: bool = False,
                  hyp: Optional[Dict] = None,
                  cache_ram: bool = False,
                  fusion: str = 'concat') -> BaseMultiModalDataset:
    """
    Dataset Factory: Instantiates the appropriate registered dataset adapter based on YAML config.

    Args:
        data_yaml_path: Path to dataset YAML file or pre-loaded dictionary
        split: 'train', 'val', or 'test'
        inputs: Modality preset or list of modality names
        img_size: Target image dimensions (H, W)
        augment: Whether to apply data augmentations
        hyp: Hyperparameters dictionary
        cache_ram: Whether to cache decoded arrays in RAM for maximum GPU saturation
        fusion: Modality fusion mode ('concat' or 'add')

    Returns:
        BaseMultiModalDataset: Concrete dataset adapter instance conforming to standard interface
    """
    if isinstance(data_yaml_path, (str, Path)):
        with open(data_yaml_path, 'r') as f:
            data_dict = yaml.safe_load(f)
    else:
        data_dict = data_yaml_path

    dataset_type = str(data_dict.get('dataset_type', 'landslide')).lower()

    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset type '{dataset_type}' not found in registry. "
            f"Available dataset types: {list(DATASET_REGISTRY.keys())}. "
            f"Register your dataset using @register_dataset('{dataset_type}')."
        )

    dataset_cls = DATASET_REGISTRY[dataset_type]

    root_path = Path(data_dict.get('path', '.'))
    split_folder = data_dict.get(split, split)
    split_dir = root_path / split_folder if not Path(split_folder).is_absolute() else Path(split_folder)
    presets = data_dict.get('presets', {})
    modality_specs = data_dict.get('modality_info', None)

    dataset_instance = dataset_cls(
        root_dir=split_dir,
        split=split,
        inputs=inputs,
        img_size=img_size,
        augment=augment,
        hyp=hyp,
        presets=presets,
        modality_specs=modality_specs,
        cache_ram=cache_ram,
        fusion=fusion
    )

    return dataset_instance


def create_dataloader(data_yaml_path, split='train', inputs='rgb_only', batch_size=8,
                      img_size=512, augment=False, hyp=None, shuffle=True, num_workers=0,
                      pin_memory=True, cache_ram=False, fusion='concat'):
    """
    Construct DataLoader for any registered dataset adapter with multi-worker prefetching optimizations.
    """
    dataset = build_dataset(
        data_yaml_path=data_yaml_path,
        split=split,
        inputs=inputs,
        img_size=img_size,
        augment=augment,
        hyp=hyp,
        cache_ram=cache_ram,
        fusion=fusion
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'drop_last': (split == 'train' and len(dataset) > batch_size),
    }

    # Enable persistent workers and prefetching when multi-threading is active
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader, dataset

