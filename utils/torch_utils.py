import os
import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator seeds for reproducibility and enable cuDNN benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True  # Auto-tune fastest convolution kernels on GPU



def select_device(device='', batch_size=None):
    """
    Select compute device (e.g. '0', '0,1', 'cpu', '').
    """
    device_str = str(device).strip().lower().replace('cuda:', '')
    cpu = device_str == 'cpu'

    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device_str:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str
        assert torch.cuda.is_available(), f"CUDA unavailable, device={device} requested"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = ' ' * (len('device') + 1)
        for i, d in enumerate(device_str.split(',') if device_str else range(n)):
            p = torch.cuda.get_device_properties(i)
            logger.info(f"{'CUDA:' + str(d) if i == 0 else space} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)")
            print(f"[Device] CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)")
        return torch.device('cuda:0')
    else:
        logger.info("Using CPU")
        print("[Device] Using CPU")
        return torch.device('cpu')


def model_info(model, verbose=False, img_size=(512, 512)):
    """Print model parameters and layers summary."""
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p:,} parameters, {n_g:,} gradients")
    return n_p, n_g



def load_pretrained_weights(model, weights_path, device='cpu'):
    """
    Load pretrained weights from a local file, URL, or official alias (e.g. 'unet_carvana').
    Automatically handles state dict key mapping from official milesial/Pytorch-UNet.

    Args:
        model: nn.Module target model
        weights_path: str, path to .pt/.pth, URL, or alias ('unet_carvana', 'official_unet', etc.)
        device: torch.device or str

    Returns:
        dict: {'matched': int, 'total': int, 'source': str}
    """
    OFFICIAL_ALIASES = {
        'unet_carvana': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth',
        'unet_carvana_1.0': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth',
        'unet_carvana_scale1.0': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth',
        'unet_carvana_0.5': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth',
        'unet_carvana_scale0.5': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth',
        'milesial/pytorch-unet': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth',
        'official_unet': 'https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth',
    }

    weights_key = str(weights_path).strip().lower()
    source_name = weights_path

    if weights_key in OFFICIAL_ALIASES:
        url = OFFICIAL_ALIASES[weights_key]
        print(f"      -> Fetching official UNet weights from: {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
        source_name = f"official UNet ({weights_key})"
    elif str(weights_path).startswith(('http://', 'https://')):
        print(f"      -> Downloading weights from: {weights_path}")
        state_dict = torch.hub.load_state_dict_from_url(weights_path, map_location=device, progress=True)
        source_name = weights_path
    elif os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
    else:
        raise FileNotFoundError(f"Pretrained weights not found: '{weights_path}'")

    # Clean out non-weight metadata if present
    if 'mask_values' in state_dict:
        state_dict.pop('mask_values')

    # Remap official milesial/Pytorch-UNet keys to our declarative Model structure
    is_milesial = any(k.startswith(('inc.', 'down1.', 'up1.')) for k in state_dict.keys())
    if is_milesial:
        milesial_mapping = {
            'inc.': 'model.1.',
            'down1.': 'model.2.',
            'down2.': 'model.3.',
            'down3.': 'model.4.',
            'down4.': 'model.5.',
            'up1.': 'model.6.',
            'up2.': 'model.7.',
            'up3.': 'model.8.',
            'up4.': 'model.9.',
            'outc.': 'model.10.',
        }
        remapped_sd = {}
        for k, v in state_dict.items():
            mapped_k = k
            for src_prefix, dst_prefix in milesial_mapping.items():
                if k.startswith(src_prefix):
                    mapped_k = dst_prefix + k[len(src_prefix):]
                    break
            remapped_sd[mapped_k] = v
        state_dict = remapped_sd

    # Transfer matching layers by name and tensor shape
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return {
        'matched': len(pretrained_dict),
        'total': len(model_dict),
        'source': source_name
    }


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False, delta=1e-4, mode='max'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode

    def __call__(self, val_score):
        score = val_score if self.mode == 'max' else -val_score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
