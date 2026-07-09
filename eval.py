import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import Generator, StyleGANGenerator


def compute_fid(real_dir: str, fake_dir: str, batch_size: int = 50, device: str = "cuda"):
    from pytorch_fid import fid_score
    return fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size=batch_size,
        device=device,
        dims=2048,
        num_workers=0,
    )


def generate_interpolation(
    generator: torch.nn.Module,
    latent_dim: int = 100,
    num_steps: int = 10,
    device: str = "cuda",
    save_path: str = None,
):
    generator.eval()

    z_start = torch.randn(1, latent_dim, device=device)
    z_end = torch.randn(1, latent_dim, device=device)

    alphas = torch.linspace(0, 1, num_steps, device=device).unsqueeze(1)
    z_interp = z_start + alphas * (z_end - z_start)

    with torch.no_grad():
        images = generator(z_interp).detach().cpu()

    images = (images + 1) / 2

    grid = vutils.make_grid(images, nrow=num_steps, padding=2)
    vutils.save_image(grid, save_path)

    return images


def generate_and_save_samples(
    generator: torch.nn.Module,
    num_samples: int = 64,
    latent_dim: int = 100,
    device: str = "cuda",
    output_dir: str = None,
):
    generator.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        images = generator(z).detach().cpu()

    images = (images + 1) / 2

    for i, img in enumerate(images):
        pil_img = vutils.to_pil_image(img)
        pil_img.save(output_dir / f"gen_{i:04d}.png")

    print(f"Saved {num_samples} generated images to {output_dir}")


def load_generator(ckpt_path: str, latent_dim: int = 100, use_stylegan: bool = False, device: str = "cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if use_stylegan:
        generator = StyleGANGenerator(latent_dim=latent_dim).to(device)
    else:
        generator = Generator(latent_dim=latent_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    print(f"Loaded generator from {ckpt_path} (epoch {ckpt['epoch']})")
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained GAN: FID score and latent interpolation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--real-dir", type=str, required=True, help="Path to folder of real CelebA images")
    parser.add_argument("--fake-dir", type=str, default=None, help="Path to folder of generated images (will generate if not provided)")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of images to generate for FID")
    parser.add_argument("--use-stylegan", action="store_true", help="Use StyleGAN-lite generator instead of DCGAN generator")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--interpolation-steps", type=int, default=10, help="Number of interpolation steps")
    parser.add_argument("--save-interpolation", type=str, default="interpolation.png", help="Path to save interpolation grid")
    parser.add_argument("--outputs-dir", type=str, default="eval_outputs", help="Directory for generated samples")
    args = parser.parse_args()

    device = args.device
    generator = load_generator(args.ckpt, args.latent_dim, args.use_stylegan, device)

    # --- Latent space interpolation ---
    print("\n--- Latent Space Interpolation ---")
    interp_images = generate_interpolation(
        generator,
        latent_dim=args.latent_dim,
        num_steps=args.interpolation_steps,
        device=device,
        save_path=args.save_interpolation,
    )
    print(f"Saved interpolation to {args.save_interpolation}")

    # --- Generate samples for FID ---
    fake_dir = Path(args.fake_dir) if args.fake_dir else Path(args.outputs_dir) / "generated"
    generate_and_save_samples(generator, args.num_samples, args.latent_dim, device, fake_dir)

    # --- Compute FID ---
    print(f"\n--- Computing FID ---")
    print(f"Real images: {args.real_dir}")
    print(f"Fake images: {fake_dir}")
    fid = compute_fid(args.real_dir, str(fake_dir), device=device)
    print(f"\nFID Score: {fid:.2f} (lower is better)")
