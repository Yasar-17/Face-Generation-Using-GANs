import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import get_dataloader
from models import Generator, Discriminator, weights_init


def gradient_penalty(D, real_images, fake_images, device):
    b_size = real_images.size(0)
    alpha = torch.rand(b_size, 1, 1, 1, device=device)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(b_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def train(
    data_dir=None,
    batch_size=64,
    latent_dim=100,
    lr_g=1e-4,
    lr_d=1e-4,
    betas=(0.0, 0.9),
    num_epochs=25,
    n_critic=5,
    lambda_gp=10.0,
    checkpoint_interval=5,
    device="cuda",
    outputs_dir="outputs",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "checkpoints").mkdir(exist_ok=True)
    (outputs_dir / "samples").mkdir(exist_ok=True)

    dataloader = get_dataloader(data_dir=data_dir, batch_size=batch_size)

    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            b_size = real_images.size(0)

            # --- Train Discriminator (n_critic times) ---
            for _ in range(n_critic):
                D.zero_grad()

                noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
                fake_images = G(noise).detach()

                output_real = D(real_images)
                output_fake = D(fake_images)

                errD_real = -output_real.mean()
                errD_fake = output_fake.mean()

                gp = gradient_penalty(D, real_images, fake_images, device)
                errD = errD_real + errD_fake + lambda_gp * gp

                errD.backward()
                optimizer_D.step()

            # --- Train Generator ---
            G.zero_grad()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_images = G(noise)
            output_gen = D(fake_images)
            errG = -output_gen.mean()
            errG.backward()
            optimizer_G.step()

            step = epoch * len(dataloader) + i

            if i % 10 == 0:
                print(
                    f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f} | "
                    f"D(x): {output_real.mean().item():.4f} | D(G(z)): {output_gen.mean().item():.4f}"
                )

        # --- Save sample grid ---
        with torch.no_grad():
            fake_grid = G(fixed_noise).detach().cpu()
        fake_grid = vutils.make_grid(fake_grid, nrow=8, normalize=True, padding=2)
        sample_path = outputs_dir / "samples" / f"epoch_{epoch:04d}.png"
        vutils.save_image(fake_grid, sample_path)

        # --- Checkpoint ---
        if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
            ckpt_path = outputs_dir / "checkpoints" / f"checkpoint_epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state": G.state_dict(),
                    "discriminator_state": D.state_dict(),
                    "optimizer_G_state": optimizer_G.state_dict(),
                    "optimizer_D_state": optimizer_D.state_dict(),
                    "fixed_noise": fixed_noise,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WGAN-GP for face generation")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to local folder of face images")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--lr-g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--n-critic", type=int, default=5, help="Number of critic updates per generator update")
    parser.add_argument("--lambda-gp", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        betas=(0.0, 0.9),
        num_epochs=args.epochs,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        outputs_dir=args.outputs_dir,
    )
