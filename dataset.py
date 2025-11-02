import os
import random
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset


class DIV2K(Dataset):
    """
    RGB images only. Generates binary mask with random square cutouts.
    sizes_max_counts: dict like {3: 11, 32: 2}
    Returns: (gt_img * mask, mask, gt_img)
    """

    def __init__(self, img_root, img_transform, sizes_max_counts, exts=(".jpg", ".jpeg", ".png"), seed=None):
        self.img_transform = img_transform
        self.sizes_max_counts = dict(sizes_max_counts)
        self.rng = random.Random(seed)

        paths = []
        for ext in exts:
            paths.extend(glob(os.path.join(img_root, f"*{ext}")))
        self.paths = sorted(paths)
        if not self.paths:
            raise RuntimeError(f"No images found in {img_root} with exts {exts}")

    def __len__(self):
        return len(self.paths)

    def _make_mask(self, h, w):
        mask = torch.ones(1, h, w, dtype=torch.float32)
        for size, max_n in self.sizes_max_counts.items():
            if size <= 0 or max_n <= 0 or size > h or size > w:
                continue
            n = self.rng.randint(0, max_n)
            for _ in range(n):
                y0 = self.rng.randint(0, h - size)
                x0 = self.rng.randint(0, w - size)
                mask[:, y0:y0 + size, x0:x0 + size] = 0.0
        return mask

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        gt_img = self.img_transform(img)
        if gt_img.dim() != 3:
            raise ValueError(f"Expected tensor CxHxW, got {gt_img.shape}")
        _, H, W = gt_img.shape

        mask = self._make_mask(H, W)
        masked = gt_img * mask
        return masked, mask, gt_img
