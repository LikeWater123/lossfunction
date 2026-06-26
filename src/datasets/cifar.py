"""CIFAR-10/100 dataset loaders for the MBA loss-family experiments.

Provides a single entry point :func:`get_cifar_loaders` returning train/test
:class:`~torch.utils.data.DataLoader` objects plus the number of classes,
along with a small :class:`Cutout` augmentation (DeVries & Taylor, 2017).

The module is CPU-only: no ``.cuda()`` calls are made here. Device placement
is the responsibility of the trainer.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Per-dataset (mean, std, num_classes) statistics for CIFAR-10/100.
_CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}

_DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "cifar100": torchvision.datasets.CIFAR100,
}


class Cutout:
    """Randomly mask out one or more square regions of an image tensor.

    Implements the Cutout augmentation of DeVries & Taylor (2017). Operates on
    a normalised ``CHW`` tensor: the masked pixels are set to zero, which is
    the per-channel mean after normalisation.

    Args:
        n_holes: Number of square holes to cut.
        length: Side length (in pixels) of each square hole.
    """

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(self.n_holes):
            cy = int(np.random.randint(h))
            cx = int(np.random.randint(w))
            y1 = int(np.clip(cy - self.length // 2, 0, h))
            y2 = int(np.clip(cy + self.length // 2, 0, h))
            x1 = int(np.clip(cx - self.length // 2, 0, w))
            x2 = int(np.clip(cx + self.length // 2, 0, w))
            mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask).to(img.dtype).unsqueeze(0)
        return img * mask.expand_as(img)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"


def _build_transform(mean, std, augment: bool, cutout: bool) -> T.Compose:
    """Build the train or test transform pipeline."""
    steps = []
    if augment:
        steps.append(T.RandomCrop(32, padding=4))
        steps.append(T.RandomHorizontalFlip(p=0.5))
    steps.append(T.ToTensor())
    steps.append(T.Normalize(mean, std))
    if augment and cutout:
        steps.append(Cutout(n_holes=1, length=16))
    return T.Compose(steps)


def get_cifar_loaders(
    name: str,
    batch_size: int = 128,
    root: str = "./data",
    num_workers: int = 4,
    augment: bool = True,
    distributed: bool = False,
    cutout: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Build CIFAR-10/100 train and test ``DataLoader`` objects.

    Args:
        name: Either ``'cifar10'`` or ``'cifar100'``.
        batch_size: Samples per batch.
        root: Directory where the dataset is (down)loaded.
        num_workers: Number of DataLoader worker processes.
        augment: If True, apply standard CIFAR training augmentation.
        distributed: If True, wrap the train set in a ``DistributedSampler``
            (assumes a torch.distributed process group is initialised).
        cutout: If True (and ``augment`` is True), append ``Cutout(1, 16)``.

    Returns:
        ``(train_loader, test_loader, num_classes)``.
    """
    key = name.lower()
    if key not in _CIFAR_STATS:
        raise ValueError(
            f"Unsupported dataset {name!r}; expected 'cifar10' or 'cifar100'."
        )
    mean, std, num_classes = _CIFAR_STATS[key]
    dataset_cls = _DATASETS[key]

    train_transform = _build_transform(mean, std, augment=augment, cutout=cutout)
    test_transform = _build_transform(mean, std, augment=False, cutout=False)

    train_set = dataset_cls(root=root, train=True, download=True, transform=train_transform)
    test_set = dataset_cls(root=root, train=False, download=True, transform=test_transform)

    train_sampler = None
    train_shuffle = True
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_shuffle = False  # the sampler owns shuffling (call set_epoch)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, num_classes


__all__ = ["Cutout", "get_cifar_loaders"]
