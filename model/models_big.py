import torch.nn as nn
import functools
import torch.nn.functional as F
import math

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
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # # 56 -> 112
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # # 112 -> 224
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
        # out = self.tanh(out)
        return out

class VGGInverterG(nn.Module):
    def __init__(self, nc=3):
        super(VGGInverterG, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            # 14 -> 28
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # 28 -> 56
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # 56 -> 112
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # 112 -> 224
            nn.ConvTranspose2d(64, nc, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        # input: (N, 100)
        out = self.conv(input)
        return out


class StudentDiscriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(StudentDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(512, d, 3, 1, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)

        self.conv5 = nn.Conv2d(d*8, d*4, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d*4)

        self.conv6 = nn.Conv2d(d*4, d*4, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(d*4)

        self.conv7 = nn.Conv2d(d*4, d*2, 3, 1, 1)
        self.conv7_bn = nn.BatchNorm2d(d*2)

        self.conv8 = nn.Conv2d(d*2, d*2, 3, 1, 1)
        self.conv8_bn = nn.BatchNorm2d(d*2)

        self.conv9 = nn.Conv2d(d*2, d, 3, 1, 1)
        self.conv9_bn = nn.BatchNorm2d(d)

        self.conv10 = nn.Conv2d(d, 1, 3, 1, 1)
        self.linear1 = nn.Linear(d*9,d)
        self.linear2 = nn.Linear(d,1)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = F.leaky_relu(self.conv7(x), 0.2)
        x = F.leaky_relu(self.conv8(x), 0.2)
        x = F.leaky_relu(self.conv9(x), 0.2)
        #x = F.leaky_relu(self.conv10(x), 0.2)

        #x = self.conv10(x)
        x = x.view(x.shape[0],-1)
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = self.linear2(x)
        #x = F.sigmoid(x)

        return x


class DeepGenerator(nn.Module):
    # initializers
    def __init__(self, nz, gen_layer=5, ngf=128, relu_out=False, tanh_out=False):
        super(DeepGenerator, self).__init__()
        norm_layer = lambda x: nn.GroupNorm(32, x)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.nz = nz
        gen_all_ch = [3, 64, 128, 256, 512, 512]
        gen_ch = gen_all_ch[gen_layer]
        num_layers = math.ceil(math.sqrt(224 // (2 ** (gen_layer - 1))))

        model = []

        in_ch = nz
        out_ch = ngf
        model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(out_ch)]
        model += [nn.LeakyReLU(0.2, True)]

        for i in range(num_layers):
            in_ch = out_ch
            out_ch = min(in_ch * 2, 512)
            model += [nn.Upsample(scale_factor=2)]
            model += [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)]
            model += [norm_layer(out_ch)]
            model += [nn.LeakyReLU(0.2, True)]

        model += [nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        model += [norm_layer(out_ch)]
        model += [nn.LeakyReLU(0.2, True)]

        padding = 0 if (gen_layer == 5) else 1
        model += [nn.Conv2d(out_ch, gen_ch, kernel_size=3, stride=1, padding=padding, bias=True)]
        if relu_out: model += [nn.ReLU()]
        if tanh_out: model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.fc = nn.Linear(self.nz, self.nz)

    # forward method
    def forward(self, input):
        l1 = self.fc(input)
        output = self.model(l1.view(-1, self.nz, 1, 1))

        return output
