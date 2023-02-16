import os
from typing import Optional, Callable

from torch.utils.data import Dataset
from os import path
from glob import glob
from PIL import Image


class SuperResolutionDataset(Dataset):
    images: list[str]  # image paths with supported extensions
    data_path: str  # valid path

    # the transformation to apply on each requested image
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    transform: Optional[Callable]

    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        assert path.exists(data_path) and path.isdir(data_path), "Data path must be a valid directory"

        self.data_path = data_path
        self.images = supported_images_in_dir(self.data_path)
        self.transform = transform

    def __getitem__(self, i: int):
        img = Image.open(self.images[i], mode='r')
        img.convert('RGB')
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.images)


EXTS = Image.registered_extensions()
SUPPORTED_EXTENSIONS = [ex[1:] if ex.startswith('.') else ex for ex, f in EXTS.items() if f in Image.OPEN]
WANTED_EXTENSIONS = [ex for ex in ['png', 'jpg', 'jpeg'] if ex in SUPPORTED_EXTENSIONS]


def supported_images_in_dir(directory: str) -> list[str]:
    paths: list[str] = []
    for ext in WANTED_EXTENSIONS:
        paths.extend(glob(path.join(directory, "**", f"*.{ext}"), recursive=True))
    return paths
