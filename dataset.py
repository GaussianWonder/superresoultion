import torch
from torch.utils.data import Dataset
import json
from os import path
from glob import glob
from PIL import Image

from math_utils import is_power_of_two


class SuperResolutionDataset(Dataset):
    images: list[str]  # image paths with supported extensions
    data_path: str  # valid path
    scaling_factor: int  # 2, 4, 8, ...
    image_count: int  # static for the lifetime of the Dataset, precalculated to not waste time on len()

    def __init__(self, data_path: str, scaling_factor: int = 2):
        assert path.exists(data_path) and path.isdir(data_path), "Data path must be a valid directory"
        assert is_power_of_two(scaling_factor), "Scaling factor must be a power of 2"

        self.data_path = data_path
        self.scaling_factor = int(scaling_factor)
        self.images = supported_images_in_dir(self.data_path)
        self.image_count = len(self.images)

    def __getitem__(self, i: int):
        img = Image.open(self.images[i], mode='r')
        img.convert('RGB')
        return None

    def __len__(self):
        return self.image_count


def supported_images_in_dir(directory: str) -> list[str]:
    supported_extensions = [ext for ext, _ in Image.EXTENSION.items()]
    return glob(path.join(
        directory,
        f"*.{'|'.join(supported_extensions)}"
    ))
