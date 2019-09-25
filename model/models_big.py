import torch.nn as nn

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