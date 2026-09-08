import random
import numpy as np
import torch


class MultiModalTransform:
    """Synchronized spatial data augmentations across multi-modal tensors and segmentation masks."""

    def __init__(self, augment=True, hyp=None, img_size=(512, 512)):
        self.augment = augment
        self.hyp = hyp or {}
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        self.fliplr_p = self.hyp.get('fliplr', 0.5) if augment else 0.0
        self.flipud_p = self.hyp.get('flipud', 0.5) if augment else 0.0
        self.rot90_p = self.hyp.get('rot90', 0.5) if augment else 0.0

    def __call__(self, multi_channel_tensor, label_mask):
        """
        Args:
            multi_channel_tensor: numpy array of shape (H, W, C)
            label_mask: numpy array of shape (H, W) or (H, W, 1)

        Returns:
            tensor: torch.FloatTensor of shape (C, H, W)
            mask: torch.FloatTensor of shape (1, H, W)
        """
        if label_mask.ndim == 2:
            label_mask = np.expand_dims(label_mask, axis=-1)

        h, w, c = multi_channel_tensor.shape

        if self.augment:
            # Horizontal flip
            if random.random() < self.fliplr_p:
                multi_channel_tensor = np.fliplr(multi_channel_tensor)
                label_mask = np.fliplr(label_mask)

            # Vertical flip
            if random.random() < self.flipud_p:
                multi_channel_tensor = np.flipud(multi_channel_tensor)
                label_mask = np.flipud(label_mask)

            # Random 90, 180, 270 degree rotation
            if random.random() < self.rot90_p:
                k = random.choice([1, 2, 3])
                multi_channel_tensor = np.rot90(multi_channel_tensor, k, (0, 1))
                label_mask = np.rot90(label_mask, k, (0, 1))

        # Convert to contiguous float32 PyTorch tensors (C, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(multi_channel_tensor).transpose(2, 0, 1)).float()
        mask = torch.from_numpy(np.ascontiguousarray(label_mask).transpose(2, 0, 1)).float()

        return tensor, mask
