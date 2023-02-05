from PIL import Image
from torch import matmul, Tensor
from torchvision.transforms.functional import to_tensor, to_pil_image

from utils.math_utils import is_power_of_two
from random import randint

from utils.pytorch_utils import IMAGENET_MEAN, IMAGENET_STD, IMAGENET_MEAN_CUDA, IMAGENET_STD_CUDA, RGB_WEIGHTS


class ImageTransform(object):
    """
    A functor which performs image transformation with the sole purpose of input unification
    The constructor defines behavior for the transformation
    An HR (high resolution image) will be cropped and then downsampled to create the LR (low resolution image)
    """
    crop_size: int
    scaling_factor: int

    def __init__(self, scaling_factor: int, crop_size: int | None):
        """
        Provide image transformation details
        :param scaling_factor: the scaling factor (pow of 2)
        :param crop_size: a predefined crop size, otherwise the largest square will be used for HR
        """
        assert is_power_of_two(scaling_factor), "Scaling factor must be a power of 2"
        assert crop_size is None or crop_size > 0, "Crop size must be undefined or greater than 0"

        self.crop_size = crop_size
        self.scaling_factor = scaling_factor

    def __call__(self, img: Image):
        hr = img.crop(self.cropping_rectangle(img.width, img.height))
        lr = hr.resize(
            (
                int(hr.width / self.scaling_factor),
                int(hr.height / self.scaling_factor)
            ),
            resample=Image.BICUBIC
        )

        assert hr.width == lr.width * self.scaling_factor and hr.height == lr.height * self.scaling_factor,\
            "Floating point error after rescaling"

        return imagenet_norm(lr, contract=False), expand_domain(resolve_to_tensor(hr))

    def cropping_rectangle(self, w: int, h: int):
        """
        Resolve the cropping rectangle params based on width and height of an image
        :param w: image width
        :param h: image height
        :return: rectangle bound of a chosen region to crop from the image
        """
        if self.crop_size is None:
            # largest centered square, with dimensions perfectly divisible by the scaling factor
            x_remainder = w % self.scaling_factor
            y_remainder = h % self.scaling_factor
            x1 = x_remainder // 2
            y1 = y_remainder // 2
            x2 = x1 + (w - x_remainder)
            y2 = y1 + (h - y_remainder)
            return x1, y1, x2, y2
        else:
            # square of size crop_size randomly positioned in the available space
            x1 = randint(1, w - self.crop_size)
            y1 = randint(1, h - self.crop_size)
            x2 = x1 + self.crop_size
            y2 = y1 + self.crop_size
            return x1, y1, x2, y2


def resolve_to_tensor(img: Image | Tensor):
    # return tensor with pixel value range of [0, 1]
    return img if isinstance(img, Tensor) else to_tensor(img)


def resolve_to_image(img: Image | Tensor):
    # return PIL.Image (pixel value range is expected to be [0, 1])
    return to_pil_image(img) if isinstance(img, Tensor) else img


def expand_domain(img_tensor: Tensor):
    # Expand pixel value range from [0, 1] to [-1, 1]
    return 2. * img_tensor - 1.


def contract_domain(img_tensor: Tensor):
    # Contract pixel value range from [-1, 1] to [0, 1]
    return (img_tensor + 1.) / 2.


def resolve_domain_space(img_tensor: Tensor, contract: bool = False):
    """
    Convert the input tensor values' range from [0, 1] | [-1, 1] to [0, 1]
    :param img_tensor: tensor with pixel value range of [0, 1] | [-1, 1]
    :param contract: True if [-1, 1], False otherwise
    :return: tensor with pixel value range of [0, 1]
    """
    return img_tensor if contract is False else contract_domain(img_tensor)


def y_luminescence(img: Image | Tensor, contract: bool = False):
    """
    Extract Y luminance channel in YCbCr color format. Used to calculate PSNR and SSIM.
    This features multi parameter input types
    :param img: Image or Tensor
    :param contract: contract pixel value range. ignored if img is Image
    :return: luminance channel Y
    """
    img_tensor = resolve_to_tensor(img)
    if isinstance(img, Tensor):
        img_tensor = resolve_domain_space(img_tensor, contract)
    return matmul(255. * img_tensor.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], RGB_WEIGHTS) / 255. + 16.


def imagenet_norm(img: Image | Tensor, contract: bool = False):
    """
    Pixel values standardized by imagenet mean and std with multi parameter input types
    :param img: Image or Tensor
    :param contract: contract pixel value range. ignored if img is Image
    :return: Tensor with pixel values standardized by imagenet mean and std
    """
    img_tensor = resolve_to_tensor(img)
    if isinstance(img, Tensor):
        img_tensor = resolve_domain_space(img_tensor, contract)

    if img_tensor.ndimension() == 3:
        return (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    elif img_tensor.ndimension() == 4:
        return (img_tensor - IMAGENET_MEAN_CUDA) / IMAGENET_STD_CUDA


def uchar_norm(img: Image | Tensor, contract: bool = False):
    # Return tensor with pixel values range normalized to uchar range [0, 255]
    img_tensor = resolve_to_tensor(img)
    if isinstance(img, Tensor):
        img_tensor = resolve_domain_space(img_tensor, contract)
    return 255. * img_tensor
