"""PyTorch dataset over the cached images."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from gymwatch.data.manifest import Record
from gymwatch.labels import EQUIPMENT, OCCUPANCY
from gymwatch.preprocess import MEAN, STD, prepare


def train_transform(image_size: tuple[int, int]) -> v2.Compose:
    h, w = image_size
    return v2.Compose(
        [
            v2.ToImage(),
            # Wide scale/ratio range mimics handheld zoom and differing camera mounts.
            v2.RandomResizedCrop((h, w), scale=(0.45, 1.0), ratio=(0.55, 0.9), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.RandomAffine(degrees=7, translate=(0.04, 0.04))], p=0.5),
            v2.RandomPerspective(distortion_scale=0.15, p=0.25),
            v2.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.3, hue=0.03),
            v2.RandomGrayscale(p=0.05),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            v2.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ]
    )


class GymDataset(Dataset):
    def __init__(self, records: Sequence[Record], image_size: tuple[int, int], train: bool):
        self.records = list(records)
        self.image_size = image_size
        self.transform = train_transform(image_size) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        r = self.records[idx]
        with Image.open(r.cache_path) as im:
            im = im.convert("RGB")
            if self.transform is not None:
                x = self.transform(im)
            else:
                x = torch.from_numpy(prepare(np.asarray(im), self.image_size))
        return x, OCCUPANCY.index(r.occupancy), EQUIPMENT.index(r.equipment)


def balanced_sampler(records: Sequence[Record], seed: int) -> WeightedRandomSampler:
    """Sample each (equipment, occupancy) class equally often per epoch."""
    keys = [(r.equipment, r.occupancy) for r in records]
    freq = Counter(keys)
    weights = torch.tensor([1.0 / freq[k] for k in keys], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True, generator=g)
