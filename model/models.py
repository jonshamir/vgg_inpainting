import torch.nn as nn

# Generator and discrimintor adapted from following repository: https://github.com/eriklindernoren/PyTorch-GAN

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.init_size = args.image_size // 4
        self.linear_layer = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

        self.conv_layers = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.linear_layer(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        image = self.conv_layers(out)
        return image


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_layers = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.ds_size = args.image_size // 2 ** 4
        self.adverse_layer = nn.Sequential(nn.Linear(128 * self.ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, image):
        out = self.conv_layers(image)
        out = out.view(out.shape[0], -1)
        validity = self.adverse_layer(out)
        return validity


class BasicGenerator(nn.Module):
    def __init__(self, input_size=100, nc=3):
        super(BasicGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 4 * 4 * 512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            # 4 -> 7
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7 -> 14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 14 -> 28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # # 28 -> 56
            nn.ConvTranspose2d(64, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # input: (N, 100)
        input = input.view(input.size(0), -1)
        out = self.fc(input)
        out = out.view(out.size(0), 512, 4, 4)
        out = self.conv(out)
        return out

class BasicDiscriminator(nn.Module):
    def __init__(self, nc=3, input_size=784):
        super(BasicDiscriminator, self).__init__()

        # 56 -> 28
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 64, 3, stride=2, padding=1))
        # 28 -> 14
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        # 14 -> 7
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        # 7 -> 4
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1))

        self.fc = nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 1))
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input: (N, nc, 56, 56)
        out = input
        out = self.conv1(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.conv2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.conv3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.conv4(out)
        out = nn.LeakyReLU(0.2)(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
