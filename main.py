#!./venv/bin/python3.10

import fire
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from dataset.dataset import SuperResolutionDataset
from dataset.transformation import ImageTransform
from model import SRResNet
from train import train
from utils.math_utils import random_string
from utils.pytorch_utils import DEVICE


class Main(object):
    @staticmethod
    def train(
            assets: str = './assets',

            crop_size: int = 96,
            scaling_factor: int = 4,

            large_kernel_size: int = 9,
            small_kernel_size: int = 3,
            n_channels: int = 64,
            n_blocks: int = 16,

            checkpoint_path: str | None = None,

            batch_size: int = 16,
            start_epoch: int = 0,
            iterations: int = 1000000,
            workers: int = 4,
            print_freq: int = 500,
            learning_rate: float = 1e-4,

            cudnn_benchmark: bool = True,
    ):
        """
        Start the training process with the given parameters
        :param assets: The assets path
        :param crop_size: Desired crop size to generate high_resolution_image
        :param scaling_factor: Desired scaling factor to downsample to a low_resolution_image
        :param large_kernel_size: Kernel size for first and last convolution blocks
        :param small_kernel_size: Kernel size for inner convolution blocks
        :param n_channels: In-Out for residual and subpixel convolution blocks
        :param n_blocks: Number of residual blocks
        :param checkpoint_path: Path to a checkpoint to load the state from
        :param batch_size: Batch size
        :param start_epoch: Start at this epoch iteration
        :param iterations: Number of training iterations
        :param workers: Data loader workers to load data
        :param print_freq: Batch frequency log printing
        :param learning_rate: Learning rate
        :param cudnn_benchmark: Set cudnn.benchmark to this value
        :return: void
        """
        cudnn.benchmark = cudnn_benchmark

        model = None
        optimizer = None
        if checkpoint_path is None:
            model = SRResNet(
                large_kernel_size=large_kernel_size,
                small_kernel_size=small_kernel_size,
                n_channels=n_channels,
                n_blocks=n_blocks,
                scaling_factor=scaling_factor,
            )
            optimizer = torch.optim.Adam(
                params=filter(lambda p: p.require_grad, model.parameters()),
                lr=learning_rate,
            )
        else:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']

        checkpoint_save = '{name}.pth'.format(name=random_string())
        print('This processing session will be saved into {file_name}'.format(file_name=checkpoint_save))
        model = model.to(DEVICE)
        criterion = nn.MSELoss().to(DEVICE)

        dataset = SuperResolutionDataset(
            data_path=assets,
            transform=ImageTransform(
                scaling_factor=scaling_factor,
                crop_size=crop_size,
            )
        )

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

        epochs = int(iterations // len(data_loader) + 1)

        for epoch in range(start_epoch, epochs):
            train(
                loader=data_loader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                print_frequency=print_freq,
            )

            torch.save(
                {
                    'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer,
                },
                checkpoint_save
            )


if __name__ == '__main__':
    fire.Fire(Main)
