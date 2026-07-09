import torch
import torch.nn as nn


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim: int = 100, w_dim: int = 512, num_layers: int = 8):
        super().__init__()
        layers = []
        layers.append(nn.Linear(latent_dim, w_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, num_features: int, w_dim: int = 512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale = nn.Linear(w_dim, num_features)
        self.style_bias = nn.Linear(w_dim, num_features)

    def forward(self, x, w):
        out = self.norm(x)
        scale = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        bias = self.style_bias(w).unsqueeze(-1).unsqueeze(-1)
        return out * (scale + 1) + bias


class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, w_dim: int = 512, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, w_dim)

        self.fc = nn.Linear(w_dim, ngf * 8 * 4 * 4)

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.adain1 = AdaIN(ngf * 4, w_dim)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.adain2 = AdaIN(ngf * 2, w_dim)

        self.conv3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.adain3 = AdaIN(ngf, w_dim)

        self.conv4 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        w = self.mapping(z)

        x = self.fc(w)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.relu(x)

        x = self.relu(self.conv1(x))
        x = self.adain1(x, w)

        x = self.relu(self.conv2(x))
        x = self.adain2(x, w)

        x = self.relu(self.conv3(x))
        x = self.adain3(x, w)

        x = self.conv4(x)
        x = self.tanh(x)
        return x


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, nc: int = 3, ndf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    latent_dim = 100
    G = Generator(latent_dim=latent_dim)
    G.apply(weights_init)
    z = torch.randn(8, latent_dim, 1, 1)
    fake = G(z)
    print(f"Generator output shape: {fake.shape}")
    print(f"Generator output range: [{fake.min():.3f}, {fake.max():.3f}]")

    print("\n--- StyleGAN-lite Generator ---")
    G_style = StyleGANGenerator(latent_dim=latent_dim)
    z = torch.randn(8, latent_dim)
    fake_style = G_style(z)
    print(f"StyleGAN Generator output shape: {fake_style.shape}")
    print(f"StyleGAN Generator output range: [{fake_style.min():.3f}, {fake_style.max():.3f}]")

    D = Discriminator()
    D.apply(weights_init)
    out_real = D(torch.randn(8, 3, 64, 64))
    out_fake = D(fake.detach())
    print(f"\nDiscriminator output shape (real): {out_real.shape}")
    print(f"Discriminator output shape (fake): {out_fake.shape}")
