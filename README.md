# Face Generation Using GANs

A PyTorch implementation of GANs for generating human faces. Uses WGAN-GP (Wasserstein GAN with Gradient Penalty) for stable training. Supports both DCGAN and StyleGAN-lite generator architectures.

## Features

- **WGAN-GP Training** - More stable than standard GAN training. Uses gradient penalty instead of weight clipping.
- **Two Generator Architectures** - Standard DCGAN generator and a StyleGAN-lite generator with a Mapping Network + AdaIN layers.
- **FID Evaluation** - Computes Frechet Inception Distance to measure generated image quality.
- **Latent Space Interpolation** - Morph between two random latent vectors to see smooth transitions in face space.
- **CelebA Support** - Automatically downloads the CelebA dataset if no local data folder is provided.
- **Custom Dataset** - Point to any folder of face images to train on your own data.

## Project Structure

```
├── train.py                  # Training script (WGAN-GP)
├── eval.py                   # Evaluation script (FID + interpolation)
├── check_images.py           # Utility to find corrupted images
├── requirements.txt          # Python dependencies
├── quick_start.bat           # Quick start guide for Windows
├── src/
│   ├── models.py             # Generator, Discriminator, StyleGANGenerator
│   ├── data.py               # Data loading (CelebA + custom folders)
│   └── visualize_batch.py    # Preview training images
└── outputs/
    ├── samples/              # Generated face grids per epoch
    └── checkpoints/          # Model checkpoints
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision, matplotlib, scipy, pytorch-fid

Install dependencies:

```
pip install -r requirements.txt
```

## Dataset

By default the code downloads the CelebA dataset (~1.5GB) on first run. You can also point to your own folder of face images:

```
python train.py --data-dir /path/to/face/images
```

## Training

```
python train.py --data-dir /path/to/faces --epochs 25 --batch-size 64
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | None | Path to folder of face images |
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr-g` | 1e-4 | Generator learning rate |
| `--lr-d` | 1e-4 | Discriminator learning rate |
| `--n-critic` | 5 | Discriminator updates per generator update |
| `--latent-dim` | 100 | Noise vector dimension |
| `--device` | cuda | Training device |

Outputs:
- `outputs/samples/epoch_XXXX.png` - Grid of generated faces after each epoch
- `outputs/checkpoints/checkpoint_epoch_XXXX.pth` - Model checkpoints

## Evaluation

```
python eval.py --ckpt outputs/checkpoints/checkpoint_epoch_0024.pth --real-dir /path/to/faces
```

This generates:
- `interpolation.png` - Latent space interpolation grid
- `eval_outputs/generated/*.png` - Generated samples for FID computation
- FID score printed to the console

## Model Architecture

**Generator (DCGAN):** Takes a 100-dimensional noise vector and upsamples through 5 transposed convolution layers to produce a 64x64 RGB image. Uses BatchNorm and ReLU after each layer, with Tanh at the output.

**StyleGAN-lite Generator:** Uses a Mapping Network to transform the latent vector into an intermediate W space, then applies AdaIN (Adaptive Instance Normalization) after each convolution layer. This gives more control over the generated features.

**Discriminator:** A standard convolutional classifier with InstanceNorm and LeakyReLU. It outputs a single score (critic output in WGAN) instead of a probability.

Both generators produce images in the range [-1, 1]. Denormalize with `(image + 1) / 2` to get [0, 1] for viewing.

## Training Details

- Uses WGAN-GP with gradient penalty coefficient lambda = 10
- Adam optimizer with betas = (0.0, 0.9)
- Images are center-cropped to 140x140 and resized to 64x64
- Pixel values normalized to [-1, 1]
- 5 critic iterations per generator update (n_critic = 5)
