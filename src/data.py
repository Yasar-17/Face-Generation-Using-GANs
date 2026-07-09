import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 0


def get_transforms():
    return transforms.Compose([
        transforms.CenterCrop(140),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


class FaceImageFolder(Dataset):
    def __init__(self, root: str, transforms=None):
        self.root = Path(root)
        self.transforms = transforms
        self.image_paths = sorted([
            p for p in self.root.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        ])
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        return img, 0


def get_dataloader(
    data_dir: str = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    download_celebA: bool = True,
):
    transforms = get_transforms()

    if data_dir and Path(data_dir).is_dir():
        dataset = FaceImageFolder(data_dir, transforms=transforms)
    else:
        try:
            dataset = datasets.CelebA(
                root="./data",
                split="train",
                target_type="attr",
                transform=transforms,
                download=download_celebA,
            )
        except Exception:
            raise RuntimeError(
                "CelebA download failed and no local data_dir was provided. "
                "Pass a valid data_dir pointing to a folder of face images."
            )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the data pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to local folder of face images")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size)
    print(f"Dataset size: {len(loader.dataset)}")
    print(f"Number of batches: {len(loader)}")

    images, _ = next(iter(loader))
    print(f"Batch shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
