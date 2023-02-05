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


class Discriminator(nn.Module):
    conv_blocks: nn.Sequential
    adaptive_pool: nn.AdaptiveAvgPool2d
    fc1: nn.Linear
    leaky_relu: nn.LeakyReLU
    fc2: nn.Linear

    def __init__(
            self,
            kernel_size: int = 3,
            # output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
            n_channels: int = 64,
            # n_blocks: number of convolutional blocks
            n_blocks: int = 8,
            # size of the first fully connected layer
            size: int = 1024
    ):
        super(Discriminator, self).__init__()

        in_channels = 3

        # A series of convolutional blocks
        # The first, third, fifth convolutional blocks increase the number of channels but retain image size
        # The second, fourth, sixth convolutional blocks retain the same number of channels but halve image size
        # The first convolutional block is unique because it does not employ batch normalization
        conv_blocks_layers = list()
        out_channels = 0    # the last out_channel value from the loop is needed later
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            conv_blocks_layers.append(
                ConvolutionalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1 if i % 2 is 0 else 2,
                    batch_norm=i is not 0,
                    activation='LeakyReLu')
            )
            in_channels = out_channels

        self.conv_blocks = nn.Sequential(*conv_blocks_layers)

        # An adaptive pool layer that resizes it to a standard size
        # For the default input size of 96 and 8 convolutional blocks, this will have no effect
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):
        # high-resolution or super-resolution images which must be classified as such
        # a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        # a score for whether it is a high-resolution image, a tensor of size (N)
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    Very Deep Convolutional Networks for Large-Scale Image Recognition, truncated such that its output
    is the 'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network'.
    Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
    Pre-trained weights of the network are used as a starting point for training on a new dataset
    The difference between the features of the high-resolution and the
    generated low-resolution images is calculated and used to train the image super-resolution model
    """
    block: nn.Sequential

    def __init__(self, i: int, j: int):
        super(TruncatedVGG19, self).__init__()

        # Load the pre-trained VGG19 available in torchvision
        vgg19 = torchvision.models.vgg19(pretrained=True)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0

        # Iterate through the convolutional section ("features") of the VGG19
        for layer in vgg19.features.children():
            truncate_at += 1

            # Count the number of maxpool layers and the convolutional layers after each maxpool
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # Break if we reach the jth convolution after the (i - 1)th maxpool
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # Check if conditions were satisfied
        assert maxpool_counter == i - 1 and conv_counter == j,\
            "One or both of i={i} and j={j} re invalid for VGG19".format(i=i, j=j)

        # Truncate to the jth convolution (+ activation) before the ith maxpool layer
        self.block = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

    def forward(self, input: Tensor):
        # input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling f, h * scaling f)
        # output: the specified VGG19 feature map
        return self.block(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)
