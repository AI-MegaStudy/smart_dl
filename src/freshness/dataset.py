from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATA_ROOT, DEFAULT_FRUIT_TYPE, DEFAULT_IMAGE_SIZE


def build_transforms(image_size: int = DEFAULT_IMAGE_SIZE, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def dataset_root(split: str, fruit_type: str = DEFAULT_FRUIT_TYPE, data_root: str | Path = DATA_ROOT) -> Path:
    return Path(data_root) / split / fruit_type


def build_dataset(
    split: str,
    fruit_type: str = DEFAULT_FRUIT_TYPE,
    data_root: str | Path = DATA_ROOT,
    image_size: int = DEFAULT_IMAGE_SIZE,
):
    root = dataset_root(split, fruit_type, data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    return datasets.ImageFolder(root=str(root), transform=build_transforms(image_size, train=(split == "train")))


def build_loader(
    split: str,
    fruit_type: str = DEFAULT_FRUIT_TYPE,
    data_root: str | Path = DATA_ROOT,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 0,
):
    dataset = build_dataset(split, fruit_type, data_root, image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=False,
    )

