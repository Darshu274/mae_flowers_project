"""Oxford Flowers-102 data loading and augmentation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
NUM_CLASSES = 102


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return ImageNet-normalised transforms for train or eval."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train, val, test) DataLoaders for Flowers-102.

    torchvision downloads the dataset automatically when not present.
    """
    data_dir = Path(data_dir)
    kwargs = dict(root=data_dir, download=True)

    train_ds = datasets.Flowers102(**kwargs, split="train", transform=get_transforms(True))
    val_ds   = datasets.Flowers102(**kwargs, split="val",   transform=get_transforms(False))
    test_ds  = datasets.Flowers102(**kwargs, split="test",  transform=get_transforms(False))

    # pin_memory only helps on CUDA; MPS doesn't support it
    pin = torch.cuda.is_available()
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin)
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kw),
        DataLoader(val_ds,   shuffle=False, **loader_kw),
        DataLoader(test_ds,  shuffle=False, **loader_kw),
    )


def make_few_shot_subset(
    dataset: Dataset,
    shots: int,
    seed: int = 42,
) -> Subset:
    """Return a Subset with exactly `shots` samples per class.

    Args:
        dataset: A Flowers102 (or any Dataset whose ._labels attribute
                 or iterating yields (image, label) pairs).
        shots: Number of images per class to keep (e.g. 1, 2, 5).
        seed: Random seed for reproducibility.

    Returns:
        torch.utils.data.Subset with shots * NUM_CLASSES samples total.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Gather indices grouped by class label
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        by_class[label].append(idx)

    selected: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        # Random permutation within each class
        perm = torch.randperm(len(indices), generator=rng).tolist()
        selected.extend(indices[perm[i]] for i in range(min(shots, len(indices))))

    return Subset(dataset, selected)


def get_few_shot_loader(
    data_dir: str | Path,
    shots: int,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Return a DataLoader over a few-shot training subset.

    The val/test loaders are unchanged; use get_dataloaders() for those.
    """
    train_ds = datasets.Flowers102(
        root=Path(data_dir), split="train", download=True,
        transform=get_transforms(True),
    )
    subset = make_few_shot_subset(train_ds, shots=shots, seed=seed)
    return DataLoader(subset, shuffle=True, batch_size=batch_size,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Flowers-102 dataset splits")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading Flowers-102 from '{args.data_dir}' (will download if needed)...\n")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=0
    )

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        ds = loader.dataset
        images, labels = next(iter(loader))
        print(f"  {name:5s}  samples={len(ds):5d}  "
              f"image shape={tuple(images.shape[1:])}  "
              f"label range=[{labels.min().item()}, {labels.max().item()}]")

    # Count unique classes across all splits
    all_labels: set[int] = set()
    for loader in (train_loader, val_loader, test_loader):
        for _, labels in loader:
            all_labels.update(labels.tolist())
    print(f"\n  Total unique classes: {len(all_labels)}")

    # Few-shot stats
    print("\nFew-shot subset sizes:")
    for shots in (1, 2, 5):
        loader = get_few_shot_loader(args.data_dir, shots=shots, num_workers=0)
        print(f"  {shots}-shot: {len(loader.dataset)} samples "
              f"({shots} × {NUM_CLASSES} classes)")
