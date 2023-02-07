from typing import Optional, Callable

from torch.utils.data import Dataset
from os import path
from glob import glob
from PIL import Image

from utils.math_utils import is_power_of_two


class SuperResolutionDataset(Dataset):
    images: list[str]  # image paths with supported extensions
    data_path: str  # valid path
    image_count: int  # static for the lifetime of the Dataset, precalculated to not waste time on len()

    # the transformation to apply on each requested image
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    transform: Optional[Callable]

    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        assert path.exists(data_path) and path.isdir(data_path), "Data path must be a valid directory"

        self.data_path = data_path
        self.images = supported_images_in_dir(self.data_path)
        self.image_count = len(self.images)
        self.transform = transform

    def __getitem__(self, i: int):
        img = Image.open(self.images[i], mode='r')
        img.convert('RGB')
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return self.image_count


def supported_images_in_dir(directory: str) -> list[str]:
    supported_extensions = [ext for ext, _ in Image.EXTENSION.items()]
    return glob(path.join(
        directory,
        f"*.{'|'.join(supported_extensions)}"
    ))
