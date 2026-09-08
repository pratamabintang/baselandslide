from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union
import torch
from torch.utils.data import Dataset
import yaml


DATASET_REGISTRY: Dict[str, Any] = {}


def register_dataset(name: str):
    """
    Decorator to register a dataset adapter or implementation into the global registry.

    Example:
        @register_dataset('my_custom_dataset')
        class MyCustomDataset(BaseMultiModalDataset):
            ...
    """
    def decorator(cls):
        DATASET_REGISTRY[name.lower()] = cls
        return cls
    return decorator


class BaseMultiModalDataset(Dataset, ABC):
    """
    Abstract Base Class and Interface for Multi-Modal Landslide Segmentation Datasets.

    Every dataset adapter MUST adhere to the following contract:
    1. `__len__() -> int`: Total number of samples in the split.
    2. `__getitem__(idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, str]`:
       - `tensor` (torch.FloatTensor): shape (C_in, H, W), float32, normalized values.
       - `mask` (torch.FloatTensor): shape (1, H, W), float32 binary mask {0.0, 1.0}.
       - `sample_id` (str): unique sample stem / filename identifier.
    3. Property `in_channels -> int`: total number of input channels for current modality setup.
    4. Property `active_inputs -> List[str]`: list of active modality keys.
    """

    def __init__(self, root_dir: Union[str, Path], split: str = 'train',
                 inputs: Union[str, List[str]] = 'rgb_only',
                 img_size: Tuple[int, int] = (512, 512),
                 augment: bool = False, hyp: Optional[Dict] = None,
                 presets: Optional[Dict] = None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.augment = augment
        self.hyp = hyp or {}
        self.presets = presets or {}

    @property
    @abstractmethod
    def in_channels(self) -> int:
        """Return the total number of channels in the output multi-modal tensor."""
        pass

    @property
    @abstractmethod
    def active_inputs(self) -> List[str]:
        """Return list of active modality names (e.g. ['IMAGE', 'DTM_NORM'])."""
        pass

    @property
    @abstractmethod
    def samples(self) -> List[str]:
        """Return list of all valid sample identifiers in this dataset split."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples in this dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Fetch sample at index.

        Returns:
            tensor (torch.FloatTensor): multi-channel tensor (C_in, H, W)
            mask (torch.FloatTensor): ground-truth binary mask (1, H, W)
            sample_id (str): identifier of the sample
        """
        pass

    def validate_sample_output(self, tensor: torch.Tensor, mask: torch.Tensor, sample_id: str):
        """
        Validation helper to check compliance with the expected interface contract.
        """
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise TypeError(f"Sample {sample_id}: tensor must be a torch.FloatTensor (got {type(tensor)} {tensor.dtype if hasattr(tensor, 'dtype') else ''})")
        if tensor.ndim != 3:
            raise ValueError(f"Sample {sample_id}: tensor must have 3 dimensions (C, H, W), got shape {tensor.shape}")
        if tensor.shape[0] != self.in_channels:
            raise ValueError(f"Sample {sample_id}: tensor channel count {tensor.shape[0]} does not match dataset.in_channels {self.in_channels}")

        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.float32:
            raise TypeError(f"Sample {sample_id}: mask must be a torch.FloatTensor (got {type(mask)})")
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError(f"Sample {sample_id}: mask must have shape (1, H, W), got shape {mask.shape}")
        if tensor.shape[1:] != mask.shape[1:]:
            raise ValueError(f"Sample {sample_id}: spatial dimensions of tensor {tensor.shape[1:]} and mask {mask.shape[1:]} do not match")
