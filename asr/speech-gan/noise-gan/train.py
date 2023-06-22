from preprocess import fbank
from model import Generator, Discriminator
from hparams import *
import torch

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


def train_epoch():
    pass


def train():
    pass


if __name__ == '__main__':
    pass
