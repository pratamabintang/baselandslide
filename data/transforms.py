import random
import numpy as np
import torch
import torch.nn.functional as F


class MultiModalTransform:
    """
    Synchronized spatial data augmentations with:
    1. Geometrically consistent directional vector transformation for Aspect (sin theta, cos theta).
    2. Guaranteed spatial dimension enforcement matching target img_size (H, W).
    """

    def __init__(self, augment=True, hyp=None, img_size=(512, 512), aspect_slice=None):
        self.augment = augment
        self.hyp = hyp or {}
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.aspect_slice = aspect_slice  # (start_channel, end_channel) for [sin_aspect, cos_aspect]

        self.fliplr_p = self.hyp.get('fliplr', 0.5) if augment else 0.0
        self.flipud_p = self.hyp.get('flipud', 0.5) if augment else 0.0
        self.rot90_p = self.hyp.get('rot90', 0.5) if augment else 0.0

    def __call__(self, multi_channel_tensor, label_mask):
        """
        Args:
            multi_channel_tensor: numpy array of shape (H, W, C)
            label_mask: numpy array of shape (H, W) or (H, W, 1)

        Returns:
            tensor: torch.FloatTensor of shape (C, img_size[0], img_size[1])
            mask: torch.FloatTensor of shape (1, img_size[0], img_size[1])
        """
        if label_mask.ndim == 2:
            label_mask = np.expand_dims(label_mask, axis=-1)

        if self.augment:
            # 1. Horizontal flip (x -> -x): spatial inversion + East-West vector inversion (sin theta -> -sin theta)
            if random.random() < self.fliplr_p:
                multi_channel_tensor = np.fliplr(multi_channel_tensor)
                label_mask = np.fliplr(label_mask)
                if self.aspect_slice is not None:
                    s, e = self.aspect_slice
                    if e - s >= 2:
                        multi_channel_tensor[:, :, s] = -multi_channel_tensor[:, :, s]

            # 2. Vertical flip (y -> -y): spatial inversion + North-South vector inversion (cos theta -> -cos theta)
            if random.random() < self.flipud_p:
                multi_channel_tensor = np.flipud(multi_channel_tensor)
                label_mask = np.flipud(label_mask)
                if self.aspect_slice is not None:
                    s, e = self.aspect_slice
                    if e - s >= 2:
                        multi_channel_tensor[:, :, s + 1] = -multi_channel_tensor[:, :, s + 1]

            # 3. Random 90, 180, 270 degree rotation: spatial rotation + azimuth vector rotation
            if random.random() < self.rot90_p:
                k = random.choice([1, 2, 3])
                multi_channel_tensor = np.rot90(multi_channel_tensor, k, (0, 1))
                label_mask = np.rot90(label_mask, k, (0, 1))
                if self.aspect_slice is not None:
                    s, e = self.aspect_slice
                    if e - s >= 2:
                        sin_val = multi_channel_tensor[:, :, s].copy()
                        cos_val = multi_channel_tensor[:, :, s + 1].copy()
                        if k == 1:    # 90 deg counter-clockwise (theta -> theta - 90 deg)
                            multi_channel_tensor[:, :, s] = -cos_val
                            multi_channel_tensor[:, :, s + 1] = sin_val
                        elif k == 2:  # 180 deg rotation (theta -> theta - 180 deg)
                            multi_channel_tensor[:, :, s] = -sin_val
                            multi_channel_tensor[:, :, s + 1] = -cos_val
                        elif k == 3:  # 270 deg counter-clockwise = 90 deg clockwise (theta -> theta + 90 deg)
                            multi_channel_tensor[:, :, s] = cos_val
                            multi_channel_tensor[:, :, s + 1] = -sin_val

        # Convert to contiguous float32 PyTorch tensors (C, H, W)
        tensor = torch.from_numpy(np.ascontiguousarray(multi_channel_tensor).transpose(2, 0, 1)).float()
        mask = torch.from_numpy(np.ascontiguousarray(label_mask).transpose(2, 0, 1)).float()

        # Enforce exact target spatial resolution (img_size[0], img_size[1])
        target_h, target_w = self.img_size
        if tensor.shape[1] != target_h or tensor.shape[2] != target_w:
            # Multi-channel continuous features: bilinear interpolation
            tensor = F.interpolate(tensor.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
            # Binary segmentation mask: nearest-neighbor interpolation to preserve {0.0, 1.0} labels
            mask = F.interpolate(mask.unsqueeze(0), size=(target_h, target_w), mode='nearest').squeeze(0)

        return tensor, mask
