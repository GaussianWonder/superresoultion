import time
from typing import TypeVar, Generic
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.modules import Module

from model import SRResNet
from utils.pytorch_utils import DEVICE


NUMERIC_T = TypeVar("NUMERIC_T", int, float)


class ValueTracker(Generic[NUMERIC_T]):
    """
    Keep track of an ever updating value
    """

    current: NUMERIC_T
    sum: NUMERIC_T
    avg: NUMERIC_T
    updates: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.updates = 0
        self.sum = 0
        self.avg = 0
        self.current = 0

    def update(self, val: NUMERIC_T, n: int = 1):
        self.updates += n
        self.current = val
        self.sum += val * n
        self.avg = self.sum / self.updates


def train(
        loader: DataLoader,
        model: SRResNet,
        criterion: Module,
        optimizer: Optimizer,
        epoch: int,
        print_frequency: int,
):
    model.train()

    processing_time = ValueTracker[float]()
    loading_time = ValueTracker[float]()
    loss_value = ValueTracker[float]()

    start_time = now()

    for i, (LRs, HRs) in enumerate(loader):
        loading_time.update(now() - start_time)

        LRs = LRs.to(DEVICE)
        HRs = HRs.to(DEVICE)

        # Generate a super resolution image from the low resolution image
        SRs = model(LRs)

        # Calculate the loss (SuperResolution from LowResolution vs HighResolution)
        loss = criterion(SRs, HRs)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # Track stats variables
        loss_value.update(loss.item(), LRs.size(0))
        processing_time.update(now() - start_time)

        # Reset start timer, to track next the batch stats
        start_time = now()

        if i % print_frequency == 0:
            print('\tEpoch {epoch}: {current}/{total}'.format(epoch=epoch, current=i, total=len(loader)))
            print(
                'Processing time: {current:.3f} avg to {avg:.3f}'.format(
                    current=processing_time.current,
                    avg=processing_time.avg
                )
            )
            print(
                'Data loading time: {current:.3f} avg to {avg:.3f}'.format(
                    current=loading_time.current,
                    avg=loading_time.avg
                )
            )
            print(
                'Loss: {current:.5f} avg to {avg:.5f}'.format(
                    current=loss_value.current,
                    avg=loss_value.avg
                )
            )

    del LRs, HRs, SRs


def now():
    return time.time()
