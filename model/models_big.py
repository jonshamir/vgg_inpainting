import torch
import torch.nn as nn
import math

VGG_NUM_CHANNELS = [3, 64, 128, 256, 512, 512]
VGG_SIZES = [224, 224, 112, 56, 28, 14]

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        # print(self.shape, input.shape)
        return input.view(*self.shape)

class VGGInverterG(nn.Module):
    def __init__(self, layer=5):
        super(VGGInverterG, self).__init__()
        nf = VGG_NUM_CHANNELS[layer] # number of feature channels

        model = [
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True)
        ]

        for _ in range(layer - 2):
            nf //= 2
            model += [
                nn.ConvTranspose2d(2 * nf, nf, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2, True)
            ]

        if (layer > 1):
            model += [nn.ConvTranspose2d(nf, 3, 4, stride=2, padding=1, bias=False)]
        else:
            model += [nn.Conv2d(nf, 3, 3, stride=1, padding=1)]
        model += [nn.Tanh()]

        self.conv = nn.Sequential(*model)

    def forward(self, input):
        # input: (N, 100)
        out = self.conv(input)
        return out


class VGGInverterD(nn.Module):
    def __init__(self, nc=3):
        super(VGGInverterD, self).__init__()
        self.conv = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(nc, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            # 112 -> 56
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            # 56 -> 28
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # 28 -> 14
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 14 -> 7
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.fc = nn.Sequential(
            # 7 -> 1
            nn.Linear(256 * 7 * 7, 512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # input: (N, nc, 224, 224)
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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
            # 28 -> 56
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 56 -> 112
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 112 -> 224
            nn.ConvTranspose2d(16, nc, 4, stride=2, padding=1, bias=False),
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

        # 224 -> 112
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(nc, 16, 3, stride=2, padding=1))
        # 112 -> 56
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(16, 32, 3, stride=2, padding=1))
        # 56 -> 28
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1))
        # 28 -> 14
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1))
        # # 14 -> 7
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1))
        # # 7 -> 4
        self.conv6 = nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1))

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
        out = self.conv5(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.conv6(out)
        out = nn.LeakyReLU(0.2)(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

class DeepGenerator(nn.Module):
    # initializers
    def __init__(self, nz=128, layer=5):
        super(DeepGenerator, self).__init__()
        final_nf = VGG_NUM_CHANNELS[layer] # number of feature channels
        nf = nz
        num_layers = 8

        self.fc = nn.Sequential(
            View(-1, nz),
            nn.Linear(nz, nf),
            View(-1, nf, 1, 1)
        )

        model = []

        for i in range(num_layers):
            if i < num_layers - layer + 1:
                model += [nn.Upsample(scale_factor=2)]

            if i == 3: # when spatial dimention is 16 make it 14 to fit with VGG sizes
                model += [nn.Conv2d(nf, nf, 3, 1, 0)]
            elif i > num_layers - 4:
                nf *= 2
                model += [nn.Conv2d(nf // 2, nf, 3, 1, 1)]
            else:
                model += [nn.Conv2d(nf, nf, 3, 1, 1)]

            model += [
                nn.BatchNorm2d(nf),
                nn.LeakyReLU(0.2)
            ]

        model += [nn.Conv2d(nf, final_nf, 1, 1, 0)]
        self.conv = nn.Sequential(*model)

    def forward(self, input):
        out = self.fc(input)
        out = self.conv(out)
        return out


class DeepDiscriminator(nn.Module):
    # initializers
    def __init__(self, layer=5):
        super(DeepDiscriminator, self).__init__()
        gen_ch = VGG_NUM_CHANNELS[layer]
        feature_size = 224 // (2**(layer - 1))
        num_layers = 8
        num_strided_layers = num_layers - layer

        model = []

        in_ch = gen_ch
        out_ch = in_ch

        if layer == 0:
            out_ch = 64
            num_strided_layers = num_layers - 1

        model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        model += [nn.LeakyReLU(0.2, True)]

        for i in range(num_strided_layers):
            in_ch = out_ch
            out_ch = min(in_ch * 2, 512)
            model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)]
            model += [nn.LeakyReLU(0.2, True)]

        for _ in range(num_layers - num_strided_layers):
            in_ch = out_ch
            out_ch = max(in_ch // 2, 128)
            model += [nn.Conv2d(in_ch, out_ch , kernel_size=3, stride=1, padding=1)]
            model += [nn.LeakyReLU(0.2, True)]

        self.model = nn.Sequential(*model)

        self.layer_size = 4 * 4 * out_ch

        self.fc = nn.Sequential(
            nn.Linear(self.layer_size, 128),
            nn.Linear(128, 1)
        )

    # forward method
    def forward(self, input):
        l1 = self.model(input)
        output = self.fc(l1.view(-1, self.layer_size))

        return output


def calc_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

class DeepEncoder(nn.Module):
    def __init__(self, layer=5):
        super(DeepEncoder, self).__init__()
        nf = VGG_NUM_CHANNELS[layer] # number of feature channels
        spatial_size = VGG_SIZES[layer]

        model = []

        for i in range(4):
            if spatial_size > 7:
                spatial_size //= 2
                model += [nn.Conv2d(nf, nf // 2, 4, stride=2, padding=1)]
            else:
                model += [nn.Conv2d(nf, nf // 2, 3, stride=1, padding=1)]

            nf //= 2
            model += [nn.LeakyReLU(0.2, True)]

        self.conv = nn.Sequential(*model)

        out_size = nf * spatial_size * spatial_size

        self.fc = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 128)
        )

    def forward(self, input):
        # input: (N, nc, 224, 224)
        out = self.conv(input)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out