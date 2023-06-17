import torch
import torch.nn as nn


class FCNormFC(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(out_features, out_features)

    def forward(self, X):
        out = self.fc1(X)
        out = self.bn(out)
        out = self.act(out)
        return self.fc2(out).T


class TransNorm(nn.Module):
    def __init__(self, in_channels, num_features, out_channels, kernel_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.act = nn.LeakyReLU()
        self.conv_trans = nn.ConvTranspose1d(in_channels, out_channels, kernel_size)

    def forward(self, X):
        out = self.act(self.bn(X))
        return self.conv_trans(out)


class Discriminator(nn.Module):
    pass


class Generator(nn.Module):
    def __init__(self, in_features, out_features, in_channels, num_features, out_channels, kernel_size, conv_nums=3):
        super().__init__()
        self.fc_norm_fc_blk = FCNormFC(in_features, out_features)
        self.trans_norm_blks = []
        for i in range(conv_nums):
            self.trans_norm_blks.append(TransNorm(in_channels[i], num_features[i], out_channels[i], kernel_size))
        self.act = nn.Tanh()

    def forward(self, X):
        out = self.fc_norm_fc_blk(X)
        for blk in self.trans_norm_blks:
            out = blk(out)
        return self.act(out).T


if __name__ == '__main__':
    X = torch.randn(8, 100)
    gen = Generator(100, 1024, [1024, 256, 128], [8, 11, 14], [256, 128, 64], 4)
    y = gen(X)
    pass
