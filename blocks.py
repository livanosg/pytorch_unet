from torch.nn import Conv2d, LeakyReLU, BatchNorm2d, Dropout2d, MaxPool2d, Softmax2d, ConvTranspose2d
from torch import cat
from torch import nn


class TwoConvs(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm2d(num_features=out_channels)
        self.drop = Dropout2d(p=dropout)
        self.conv_1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu_1 = LeakyReLU(negative_slope=0.1)
        self.bn_1 = BatchNorm2d(num_features=out_channels)
        self.drop_1 = Dropout2d(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.bn_1(x)
        x = self.drop_1(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.two_conv = TwoConvs(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.two_conv(x)
        x = self.pool(skip)
        return x, skip


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, branch):
        super().__init__()
        self.branch = branch
        self.trans = ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=2, stride=2, padding=0)
        self.relu = LeakyReLU(negative_slope=0.1)
        self.bn = BatchNorm2d(out_channels)
        self.drop = Dropout2d(p=dropout)
        self.two_convs = TwoConvs(in_channels=in_channels, out_channels=out_channels, dropout=dropout)

    def forward(self, x, skip_connection):
        x = self.trans(x)
        x = self.relu(x)
        x = self.bn(x)
        skip_connection_2 = self.drop(x)
        x = cat([skip_connection_2, skip_connection], dim=1)
        x = self.two_convs(x)
        if self.branch == 1:
            return x, skip_connection_2
        if self.branch == 2:
            return x


class Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.block_1 = DownSampling(1, 64, dropout)
        self.block_2 = DownSampling(64, 128, dropout)
        self.block_3 = DownSampling(128, 256, dropout)
        self.block_4 = DownSampling(256, 512, dropout)
        self.bridge = TwoConvs(512, 1024, dropout)
        # self.block_1 = DownSampling(1, 32, dropout)
        # self.block_2 = DownSampling(32, 64, dropout)
        # self.block_3 = DownSampling(64, 128, dropout)
        # self.block_4 = DownSampling(128, 256, dropout)
        # self.bridge = TwoConvs(256, 512, dropout)

    def forward(self, x):  # x  skip
        x, skip_1 = self.block_1(x)  # 256, 512
        x, skip_2 = self.block_2(x)  # 128, 256
        x, skip_3 = self.block_3(x)  # 64, 128
        x, skip_4 = self.block_4(x)  # 32,  64
        x = self.bridge(x)
        return x, skip_1, skip_2, skip_3, skip_4


class Decoder(nn.Module):
    def __init__(self, dropout, branch, output_classes):
        super().__init__()
        self.branch = branch
        self.block_1 = UpSampling(1024, 512, dropout=dropout, branch=self.branch)
        self.block_2 = UpSampling(512, 256, dropout=dropout, branch=self.branch)
        self.block_3 = UpSampling(256, 128, dropout=dropout, branch=self.branch)
        self.block_4 = UpSampling(128, 64, dropout=dropout, branch=self.branch)
        self.output = Output(64, output_classes)
        # self.block_1 = UpSampling(512, 256, dropout=dropout, branch=self.branch)
        # self.block_2 = UpSampling(256, 128, dropout=dropout, branch=self.branch)
        # self.block_3 = UpSampling(128, 64, dropout=dropout, branch=self.branch)
        # self.block_4 = UpSampling(64, 32, dropout=dropout, branch=self.branch)
        # self.output = Output(32, output_classes)

    def forward(self, x, skip_1, skip_2, skip_3, skip_4):
        if self.branch == 1:
            x, skip_21 = self.block_1(x, skip_4)  # 64,  64
            x, skip_22 = self.block_2(x, skip_3)  # 128, 128
            x, skip_23 = self.block_3(x, skip_2)  # 256, 256
            x, skip_24 = self.block_4(x, skip_1)  # 512, 512
            x = self.output(x)
            return x, skip_21, skip_22, skip_23, skip_24

        if self.branch == 2:
            x = self.block_1(x, skip_4)
            x = self.block_2(x, skip_3)
            x = self.block_3(x, skip_2)
            x = self.block_4(x, skip_1)
            x = self.output(x)
            return x


class Output(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.output = Conv2d(in_channels=in_channels, out_channels=classes, kernel_size=1)
        self.softmax = Softmax2d()

    def forward(self, x):
        x = self.output(x)
        x = self.softmax(x)
        return x


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Output(in_channels=1, classes=2)

    def forward(self, x):
        output = self.encoder(x)
        return output
