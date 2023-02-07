import torchvision
from torch import nn, Tensor, load

import math
from utils.math_utils import is_power_of_two


class ConvolutionalBlock(nn.Module):
    block: nn.Sequential

    def __init__(
            self,
            # Conv2d params
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            # include BatchNorm2d or not
            batch_norm: bool = False,
            # PReLU | LeakyReLU | Tanh
            activation: str | None = None
    ):
        # Just a power upped Conv2D
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        layers = list()

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            )
        )

        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        # (N, out_channels, w, h) image or tensor
        return self.block(input)


class SubPixelConvolutionalBlock(nn.Module):
    block: nn.Sequential

    def __init__(
            self,
            kernel_size: int = 3,
            n_channels: int = 64,
            scaling_factor: int = 2
    ):
        assert is_power_of_two(scaling_factor), "Scaling factor must be a power of 2"

        # Convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
        super(SubPixelConvolutionalBlock, self).__init__()

        layers = list()

        layers.append(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels * (scaling_factor ** 2),
                kernel_size=kernel_size, padding=kernel_size // 2
            )
        )

        layers.append(nn.PixelShuffle(upscale_factor=scaling_factor))
        layers.append(nn.PReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        # convolution: (N, n_channels * scaling factor^2, w, h) image or Tensor
        # shuffle: (N, n_channels, w * scaling factor, h * scaling factor)
        # PReLU: (N, n_channels, w * scaling factor, h * scaling factor)
        return self.block(input)


class ResidualBlock(nn.Module):
    block: nn.Sequential

    def __init__(self, kernel_size: int = 3, n_channels: int = 64):
        # two convolutional blocks with a residual connection across them.
        # reduces the vanishing gradients problem
        super(ResidualBlock, self).__init__()

        layers = list()

        layers.append(
            ConvolutionalBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                batch_norm=True,
                activation='PReLu'
            )
        )

        layers.append(
            ConvolutionalBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                batch_norm=True,
                activation=None
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, input: Tensor):
        # forward (N, n_channels, w, h) -> (N, n_channels, w, h)
        residual = input
        output = self.block(input)
        return output + residual


class SRResNet(nn.Module):
    conv1: ConvolutionalBlock
    residual: nn.Sequential
    conv2: ConvolutionalBlock
    upscale: nn.Sequential
    final: ConvolutionalBlock

    def __init__(
            self,
            # kernel size of the first and last convolutions which transform the inputs and outputs
            large_kernel_size: int = 9,
            # kernel size of all convolutions in-between
            small_kernel_size: int = 3,
            # number of channels in-between
            n_channels: int = 64,
            # number of residual blocks
            n_blocks: int = 16,
            # factor to scale input images by (along both dimensions) in the subpixel convolutional block (pow of 2)
            scaling_factor: int = 4
    ):
        assert is_power_of_two(scaling_factor), "Scaling factor must be a power of 2"

        # The super image resolution generator
        super(SRResNet, self).__init__()

        # The first convolutional block
        self.conv1 = ConvolutionalBlock(
            in_channels=3,
            out_channels=n_channels,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation='PReLu'
        )

        # A sequence of n_blocks residual blocks, each containing a skip-connection across the block
        self.residual = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for _ in range(n_blocks)]
        )

        self.conv2 = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=small_kernel_size,
            batch_norm=True,
            activation=None
        )

        # Subpixel convolution by a factor of 2 until scaling_factor is reached
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.upscale = nn.Sequential(
            *[
                SubPixelConvolutionalBlock(
                    kernel_size=small_kernel_size,
                    n_channels=n_channels,
                    scaling_factor=2
                ) for _ in range(n_subpixel_convolution_blocks)
            ]
        )

        # The last convolutional block
        self.final = ConvolutionalBlock(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=large_kernel_size,
            batch_norm=False,
            activation='Tanh'
        )

    def forward(self, lr_imgs):
        # low-resolution input images, a tensor of size (N, 3, w, h)
        # to
        # super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        output = self.conv1(lr_imgs)  # (N, 3, w, h)
        residual_t = output  # (N, n_channels, w, h)
        output = self.residual(output)  # (N, n_channels, w, h)
        output = self.conv2(output)  # (N, n_channels, w, h)
        output = output + residual_t  # (N, n_channels, w, h)
        output = self.upscale(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        return self.final(output)  # (N, 3, w * scaling factor, h * scaling factor)

    def load(self, to_load):
        mappings = load(to_load)['model']
        self.load_state_dict(mappings)
