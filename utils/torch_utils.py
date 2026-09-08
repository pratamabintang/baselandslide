import os
import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def init_seeds(seed=0):
    """Initialize random number generator seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
