import argparse
import math

import matplotlib.pyplot as plt
import torch

from data import get_dataloader


def unnormalize(tensor):
    return tensor * 0.5 + 0.5


def visualize_batch(dataloader, num_images: int = 64, save_path: str = None):
    images, _ = next(iter(dataloader))
    images = images[:num_images]

    grid_size = int(math.ceil(math.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            img = unnormalize(images[i].permute(1, 2, 0)).clamp(0, 1)
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout(pad=0.5)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved grid to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a batch from the data pipeline")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to local folder of face images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-images", type=int, default=64, help="Number of images to display in the grid")
    parser.add_argument("--save", type=str, default=None, help="Path to save the output image")
    args = parser.parse_args()

    loader = get_dataloader(data_dir=args.data_dir, batch_size=args.batch_size)
    visualize_batch(loader, num_images=args.num_images, save_path=args.save)
