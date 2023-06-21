from dataclasses import dataclass, field
from typing import List


@dataclass
class GeneratorHParams:
    in_features: int = field(default=100)
    hid_features: int = field(default=1024)
    out_features: int = field(default=1024)
    in_channels: List[int] = field(default_factory=lambda: [1024, 256, 128])
    num_features: List[int] = field(default_factory=lambda: [8, 11, 14])
    out_channels: List[int] = field(default_factory=lambda: [256,128,64])
    kernel_size: int = field(default=4)
    conv_num: int = field(default=3)


@dataclass
class DiscriminatorHParams:
    in_features: int = field(default=1024)
    hid_features: int = field(default=1024)
    out_features: int = field(default=1024)
    in_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_features: List[int] = field(default_factory=lambda: [14, 11, 8])
    out_channels: List[int] = field(default_factory=lambda: [128, 256, 1024])
    kernel_size: int = field(default=4)
    conv_num: int = field(default=3)
