import io
import pathlib
from typing import Optional, TypeVar, Union

import cv2
import einops
import imageio
import numpy as np
import PIL
import torch
import torchvision

from .base import BaseZipSaver

# --------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------

# path types: str or pathlib.Path
_Path = TypeVar("_Path", str, pathlib.Path)

# Image types: numpy ndarray to torch Tensor
_Image = TypeVar("_Image", torch.Tensor, np.ndarray)

# --------------------------------------------------------------------------
# torch to numpy conversions and vice versa
# --------------------------------------------------------------------------


def image_to_tensor(image: np.ndarray, keepdim=False) -> torch.Tensor:
    """Convert a batch of images in the numpy format (B, ..., C) to the Torch format (B, C, ...). In other words, it converts the channel-last data format to the channel-first data format.

    Args:
        image (np.ndarray): input images in the numpy format
        keepdim (bool, optional): if set to True, this function keeps the batch dimension when the batch size is 1.

    Raises:
        ValueError: when the input is not 2d, 3d, or 4d.

    Returns:
        torch.Tensor: converted image according to the PyTorch format (B, C, ...).
    """
    image_torch = torch.from_numpy(image)
    input_shape = image.shape

    if len(input_shape) == 2:
        # HW --> CHW
        image_torch = image_torch.unsqueeze(0)
    elif len(input_shape) == 3:
        # HWC --> CHW
        image_torch = image_torch.permute(2, 0, 1)
    elif len(input_shape) == 4:
        # BHWC --> BCHW
        image_torch = image_torch.permute(0, 3, 1, 2)
        keepdim = True
    else:
        raise ValueError("image_to_tensor only supports 2d, 3d, or 4d inputs")

    # adding the batch dimension if needed
    if not keepdim:
        return image_torch.unsqueeze(0)
    else:
        return image_torch


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    """Converts the input images given in the torch format (B, C, ...) to the numpy format (B, ..., C). In other words, it converts the channel-first data format to the channel-last data format. It moves the tensors to CPU if necessary and removes the batch dimension when the batch-size is 1.

    Args:
        image (torch.Tensor): a batch of images in the torch format (B, C, ...).

    Raises:
        ValueError: when the input is not 2d, 3d, or 4d.

    Returns:
        np.ndarray: converted image according to the numpy format (B, ..., C).
    """
    image_np: np.ndarray = image.squeeze().detach().cpu().numpy()
    input_shape = image_np.shape

    if len(input_shape) == 2:
        # no need to change gray scale images
        return image_np
    elif len(input_shape) == 3:
        # CHW --> HWC
        return image_np.transpose(1, 2, 0)
    elif len(input_shape) == 4:
        # BCHW --> BHWC
        return image_np.transpose(0, 2, 3, 1)
    else:
        raise ValueError("tensor_to_image only supports 2d, 3d, or 4d inputs")


# --------------------------------------------------------------------------
# Image io utilities
# --------------------------------------------------------------------------


def load_image(fpath: _Path, mode: str = "RGB") -> np.ndarray:
    """Load image from disc into RGB or BGR format.

    Args:
        fpath (_Path): Path to the image
        mode (str, optional): determines the color format of the loaded image. Either "RGB" or "BGR". Defaults to "RGB".

    Raises:
        ValueError: when the mode is not in ["RGB" | "BGR"].

    Returns:
        np.ndarray: loaded image in the given color format.
    """
    if not mode in ["RGB", "BGR"]:
        raise ValueError(f"{mode} is not supported")
    image = cv2.imread(str(fpath))
    if mode == "RGB":
        image = from_cv2(image)
    return image


def save_image(fpath: _Path, image: np.ndarray, input_mode: str = "RGB") -> None:
    if not input_mode.upper() in ["RGB", "BGR", "L"]:
        raise ValueError(f"{input_mode} is not supported")
    if input_mode == "RGB":
        image = to_cv2(image)
    cv2.imwrite(str(fpath), image)


def save_tensor_image(fpath: _Path, image: torch.Tensor, drange: tuple = (0, 255)) -> None:
    image = from_torch(image, drange)
    save_image(fpath, image)


def make_grid(img, grid_size):
    gw, gh = grid_size
    _N, H, W, C = img.shape
    img = img.reshape(gh, gw, H, W, C)
    img = img.transpose(0, 2, 1, 3, 4)
    img = img.reshape(gh * H, gw * W, C)
    return img


def save_image_grid(fpath, imgs, grid_size, drange=(0, 255)):
    grid = torchvision.utils.make_grid(imgs, padding=0, nrow=grid_size[0])
    grid = from_torch(grid, drange)
    save_image(fpath, grid)


def image_to_bytes(image: np.ndarray, mode: str = "RGB") -> bytes:
    if mode.upper() == "RGB":
        image = to_cv2(image)
    success, buffer = cv2.imencode(".png", image)
    if success:
        image_bytes = io.BytesIO(buffer)
    else:
        raise Exception("not able to convert the image to binary")
    return image_bytes.getvalue()


class ImageZipSaver(BaseZipSaver):
    "interface for saving a list of images to a zipfile incrementally"

    def __init__(self, fpath: Union[str, pathlib.Path], basename: Optional[str] = "data") -> None:
        super().__init__(fpath, basename=basename)

    def save_file(self, image: np.ndarray) -> None:
        image_bytes = image_to_bytes(image)
        self.zip_file.writestr(self.get_current_name() + ".png", image_bytes)


# --------------------------------------------------------------------------
# Conversion utilities
# --------------------------------------------------------------------------


def from_torch(image: torch.Tensor, drange: tuple = (0, 255)) -> np.ndarray:
    """Take a batch of torch images with pixel values in [drange(0), drange(1)] as input and convert them to uint8 numpy images with channel-last data format and pixel valyes in [0, 255].

    Args:
        image (torch.Tensor): input torch images with normalized valye
        drange (tuple, optional): _description_. Defaults to (0, 255).

    Returns:
        np.ndarray: the output image in the numpy format with dtype = uint8.
    """
    img = tensor_to_image(image)
    img = change_drange(img, drange[0], drange[1])
    return to_uint8(img)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert images with arbitrary data type (for example floating point) to uint8 images.

    Args:
        image (np.ndarray): input image with arbitrary dtype

    Returns:
        np.ndarray: output image with dtype = uint8
    """
    image = np.rint(image).clip(0, 255).astype("uint8")
    return image


def to_cv2(image: np.ndarray) -> np.ndarray:
    """Convert the RGB image to the opencv BGR image

    Args:
        image (np.ndarray): input image in the RGB format

    Returns:
        np.ndarray: image in the BGR format
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def from_cv2(image: np.ndarray) -> np.ndarray:
    """Convert the opencv BGR format to an RGB image

    Args:
        image (np.ndarray): input image in the BGR format

    Returns:
        np.ndarray: image in the RGB format
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def change_drange(image: _Image, dlow: int = 0, dhigh: int = 255) -> _Image:
    """Get an image with the pixel values in [dlow, dhigh] and re-normalize it to [0, 255]

    Args:
        image (_Image): input image
        dlow (int, optional): the minimum value of the input pixels. Defaults to 0.
        dhigh (int, optional): the maximum value of the input pixels. Defaults to 255.

    Returns:
        _Image: re-normalized image
    """
    return (image - dlow) / (dhigh - dlow) * 255


# --------------------------------------------------------------------------


def gen_lut():
    """
    Generate a label colormap compatible with opencv lookup table, based on
    Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
    appendix C2 `Pseudocolor Generation`.
    :Returns:
        color_lut : opencv compatible color lookup table
    """
    tobits = lambda x, o: np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)
    arr = np.arange(256)
    r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
    g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
    b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
    return np.concatenate([[[b]], [[g]], [[r]]]).T


def labels2rgb(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Convert a label image to an rgb image using a lookup table
    :Parameters:
        labels : an image of type np.uint8 2D array
        lut : a lookup table of shape (256, 3) and type np.uint8
    :Returns:
        colorized_labels : a colorized label image
    """
    return cv2.LUT(cv2.merge((labels, labels, labels)), lut)


def save_mim(image_list, fapth, fps=60):
    imageio.mimsave(fapth, image_list, fps=fps)
