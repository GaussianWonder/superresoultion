#!./venv/bin/python3.10

import fire
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset.dataset import SuperResolutionDataset
from dataset.transformation import ImageTransform, y_luminescence, resolve_to_image, contract_domain, imagenet_norm
from model import SRResNet
from train import train, ValueTracker, now
from utils.math_utils import random_string
from utils.pytorch_utils import DEVICE
import os


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

            batch_size: int = 32,
            start_epoch: int = 0,
            iterations: int = 1000,
            workers: int = 4,
            print_freq: int = 100,
            learning_rate: float = 0.001,

            cudnn_benchmark: bool = False,
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
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            model = SRResNet(
                large_kernel_size=large_kernel_size,
                small_kernel_size=small_kernel_size,
                n_channels=n_channels,
                n_blocks=n_blocks,
                scaling_factor=scaling_factor,
            )
            optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
            )
        else:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            model = checkpoint['model']
            optimizer = checkpoint['optimizer']

        checkpoint_save = checkpoint_path if checkpoint_path is not None else '{name}.pth'.format(name=random_string())
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
        print(
            "{iterations} iterations from a total of {count} batches in {total} epochs".format(
                iterations=iterations,
                count=len(data_loader),
                total=epochs,
            )
        )

        batch_start = now()
        batch_time = ValueTracker[float]()
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

            batch_time.update(now() - batch_start)
            batch_start = now()
            # TODO make this print statement a function inside the value tracker
            print(
                '\tEpoch total time: {current:.3f} avg to {avg:.3f}'.format(
                    current=batch_time.current,
                    avg=batch_time.avg
                )
            )
            print('')

    @staticmethod
    def evaluate(
            checkpoint_path: str,
            assets: str = './assets',

            scaling_factor: int = 4,

            workers: int = 4,
    ):
        dataset = SuperResolutionDataset(
            data_path=assets,
            transform=ImageTransform(
                scaling_factor=scaling_factor,
                crop_size=None,
            )
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)

        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model']
        model = model.to(DEVICE)

        SSs = ValueTracker[float]()
        PSNRs = ValueTracker[float]()

        with torch.no_grad():
            for i, (LRs, HRs) in enumerate(loader):
                LRs = LRs.to(DEVICE)  # (1, 3, w / 4, h / 4), imagenet-normed
                HRs = HRs.to(DEVICE)  # (1, 3, w, h), in [-1, 1]

                SRs = model(LRs)  # (1, 3, w, h), in [-1, 1]

                SRy = y_luminescence(img=SRs, contract=True)
                HRy = y_luminescence(img=HRs, contract=True)

                PSNR = peak_signal_noise_ratio(
                    HRy.cpu().numpy(),
                    SRy.cpu().numpy(),
                    data_range=255.
                )
                SS = structural_similarity(
                    HRy.cpu().numpy(),
                    SRy.cpu().numpy(),
                    data_range=255.

                )

                PSNRs.update(PSNR, LRs.size(0))
                SSs.update(SS, LRs.size(0))

    @staticmethod
    def compare(
            checkpoint_path: str,
            assets: str = './assets',

            scaling_factor: int = 4,

            workers: int = 4,
            index: int = 0,
    ):
        dataset = SuperResolutionDataset(
            data_path=assets,
            transform=ImageTransform(
                scaling_factor=scaling_factor,
                crop_size=None,
            )
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)

        assert index < len(dataset), "Index {idx} is not available in the dataset".format(idx=index)

        checkpoint = torch.load(checkpoint_path)
        model = checkpoint['model'].to(DEVICE)

        with torch.no_grad():
            for i, (_, HRs) in enumerate(loader):
                if i == index:
                    HRs = HRs.to(DEVICE)  # (1, 3, w, h), in [-1, 1]

                    init_pixel_value = contract_domain(HRs)  # in [0, 1]
                    imagenet_normalized = imagenet_norm(init_pixel_value)  # imagenet normed

                    SRs = model(imagenet_normalized)

                    generated_mapped = contract_domain(SRs.squeeze(0))
                    generated_image = resolve_to_image(generated_mapped)

                    generated_image.show()


if __name__ == '__main__':
    fire.Fire(Main)
