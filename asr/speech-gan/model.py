import torch
import torch.nn as nn

from hparams import *


class FCNormFC(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hid_features)
        self.bn = nn.BatchNorm1d(hid_features)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(hid_features, out_features)

    def forward(self, X):
        out = self.fc1(X)
        out = self.bn(out)
        out = self.act(out)
        return self.fc2(out).T


class TransConvNorm(nn.Module):
    def __init__(self, in_channels, num_features, out_channels, kernel_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.conv_trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)

    def forward(self, X):
        out = self.act(self.bn(X))
        return self.conv_trans(out)


class ConvNorm(nn.Module):
    def __init__(self, in_channels, num_features, out_channels, kernel_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, X):
        out = self.conv(X)
        return self.act(self.bn(out))


class Discriminator(nn.Module):
    def __init__(self,
                 in_features: int,
                 hid_features: int,
                 out_features: int,
                 in_channels: List[int],
                 num_features: List[int],
                 out_channels: List[int],
                 kernel_size: int,
                 conv_nums: int):
        super().__init__()
        assert conv_nums == len(in_channels) == len(num_features) == len(out_channels)
        self.fc_norm_fc_blk = FCNormFC(in_features, hid_features, out_features)
        self.conv_blks = []
        for i in range(conv_nums):
            self.conv_blks.append(ConvNorm(in_channels[i], num_features[i], out_channels[i], kernel_size))
        self.act = nn.Sigmoid()

    def forward(self, X):
        out = X
        for blk in self.conv_blks:
            out = blk(out)
        out = self.fc_norm_fc_blk(out.T)
        return self.act(out)


class Generator(nn.Module):
    def __init__(self,
                 in_features: int,
                 hid_features: int,
                 out_features: int,
                 in_channels: List[int],
                 num_features: List[int],
                 out_channels: List[int],
                 kernel_size: int,
                 conv_nums: int):
        super().__init__()
        assert conv_nums == len(in_channels) == len(num_features) == len(out_channels)
        self.fc_norm_fc_blk = FCNormFC(in_features, hid_features, out_features)
        self.trans_conv_norm_blks = []
        for i in range(conv_nums):
            self.trans_conv_norm_blks.append(
                TransConvNorm(in_channels[i], num_features[i], out_channels[i], kernel_size))
        self.act = nn.Tanh()

    def forward(self, X):
        out = self.fc_norm_fc_blk(X)
        for blk in self.trans_conv_norm_blks:
            out = blk(out)
        return self.act(out)


if __name__ == '__main__':
    X = torch.randn(8, 8, 100)
    gen_hparam = GeneratorHParams()
    dis_hparam = DiscriminatorHParams()
    gen = Generator(
        gen_hparam.in_features,
        gen_hparam.hid_features,
        gen_hparam.out_features,
        gen_hparam.in_channels,
        gen_hparam.num_features,
        gen_hparam.out_channels,
        gen_hparam.kernel_size,
        gen_hparam.conv_num
    )
    y = gen(X)
    dis = Discriminator(
        dis_hparam.in_features,
        dis_hparam.hid_features,
        dis_hparam.out_features,
        dis_hparam.in_channels,
        dis_hparam.num_features,
        dis_hparam.out_channels,
        dis_hparam.kernel_size,
        dis_hparam.conv_num
    )
    z = dis(y).T
    pass