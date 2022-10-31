import abc
import io
import os
import pathlib
import shutil
import sys
import zipfile
from typing import Any, List, Optional, TypeVar, Union, Callable, Tuple
import functools
import time
import datetime
import string
import random

import cv2
import imageio
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------------------------
# types
# --------------------------------------------------------------------------

_ImageType = TypeVar("_ImageType")
_PathLike = TypeVar("_PathLike")


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

IMG_EXTENSIONS = {
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
    ".webp",
}

# --------------------------------------------------------------------------
# for logging
# --------------------------------------------------------------------------


def get_date_uid():
    """
    Generate a unique id based on date.
    """
    cur_time = str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))
    random_id = "".join(random.choices(string.digits + string.ascii_lowercase, k=7))
    return f"{cur_time}-{random_id}"


# --------------------------------------------------------------------------
# tqdm utils
# --------------------------------------------------------------------------


def tqdm_setup(iterable, should_enumerate=False):
    if should_enumerate:
        return tqdm(enumerate(iterable), total=len(iterable), leave=False)
    else:
        return tqdm(iterable, total=len(iterable), leave=False)


# --------------------------------------------------------------------------
# torch utils
# --------------------------------------------------------------------------


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def image_to_vid(images, path):
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(path, imgs)


def save_image(fpath: _PathLike, image: _ImageType, drange: Tuple[int, int] = (0, 255)):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    low, high = drange
    img = (image + low) / (high - low) * 255
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    if img.ndim == 4:
        img = img[0]
    img = img.transpose(1, 2, 0)
    imageio.imwrite(fpath, img)


# --------------------------------------------------------------------------
# image utils
# --------------------------------------------------------------------------


def load_img_from_bytes(imge_bytes, mode="cv2"):
    if mode == "cv2":
        nparr = np.frombuffer(imge_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


# --------------------------------------------------------------------------
# file utils
# --------------------------------------------------------------------------


def relative_path(path, part_num):
    """
    extract the relative path by removing the first part_num parts in the given address
    """
    path_ = pathlib.Path(path)
    path_ = path_.relative_to(*path_.parts[0:part_num])
    return path_


def make_dir(path, remove=True):
    path = pathlib.Path(path)
    if remove and path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)


def find_all_files(directory, format="*.*"):
    "find all files recursively in a directory that matches the given format"
    dirpath = pathlib.Path(directory)
    assert dirpath.is_dir()
    file_lists = dirpath.rglob(format)
    return file_lists


def merge_directory(dir1, dir2, dest):
    """
    copy all files in dir1 and dir2 to the target directoy dest.
    """
    counter = 0
    final_dir = pathlib.Path(dest)
    utils.make_dir(final_dir)

    file_list1 = utils.find_all_files(dir1)
    file_list2 = utils.find_all_files(dir1)

    for f in sorted(file_list1):
        shutil.copy(f, final_dir.joinpath(f"sample{counter:06d}.png"))
        counter += 1

    for f in file_list2:
        shutil.copy(f, final_dir.joinpath(f"sample{counter:06d}.png"))
        counter += 1


class Logger(object):
    """Redirect stderr to stdout, optionally print stdout to a file, and optionally force flushing on both stdout and the file."""

    def __init__(
        self, file_name: str = None, file_mode: str = "w", should_flush: bool = True
    ):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: Union[str, bytes]) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if isinstance(text, bytes):
            text = text.decode()
        if (
            len(text) == 0
        ):  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()
            self.file = None


def is_image(fpath):
    result = True if get_ext(fpath) in IMG_EXTENSIONS else False
    return result


def get_ext(fpath):
    return pathlib.Path(str(fpath)).suffix


def get_name(fpath):
    return pathlib.Path(str(fpath)).stem


class BaseZipSaver(abc.ABC):
    def __init__(
        self, fpath: Union[str, pathlib.Path], basename: Optional[str] = "data"
    ) -> None:
        super().__init__()
        self.zip_file = zipfile.ZipFile(fpath, mode="w")
        self.basename = basename
        self.counter = 0

    def __del__(self):
        try:
            self.zip_file.close()
        finally:
            self.zip_file = None

    @abc.abstractmethod
    def save_file(self, file: Any, **kwargs):
        pass

    def save_file_list(self, file_list: Any, **kwargs):
        for file in file_list:
            self.save_file(file, **kwargs)

    def get_current_name(self):
        name = f"{self.basename}/sample{self.counter:06d}"
        self.counter += 1
        return name


class ImageZipSaver(BaseZipSaver):
    "interface for saving a list of images to a zipfile incrementally"

    def __init__(
        self,
        fpath: Union[str, pathlib.Path],
        basename: Optional[str] = "data",
        save_lib: Optional[str] = "cv2",
    ) -> None:
        super().__init__(fpath, basename=basename)
        self.save_lib = save_lib

    def _image_to_bytes(
        self,
        image: _ImageType,
        mode: Optional[str] = "cv2",
        ext: Optional[str] = ".png",
    ) -> bytes:
        if ext not in IMG_EXTENSIONS:
            raise IOError("not accepted Image extention")

        if self.save_lib == "cv2":
            success, buffer = cv2.imencode(".png", image)
            if success:
                image_bytes = io.BytesIO(buffer)
            else:
                raise Exception("not abale to convert the image to binary")

        elif self.save_lib == "pil":
            image_bytes = io.BytesIO()
            image.save(image_bytes, "PNG")

        return image_bytes.getvalue()

    def save_file(self, file: _ImageType) -> None:
        image_bytes = self._image_to_bytes(file)
        self.zip_file.writestr(self.get_current_name() + ".png", image_bytes)


class TensorZipSaver(BaseZipSaver):
    def __init__(
        self, fpath: Union[str, pathlib.Path], basename: Optional[str] = "data"
    ) -> None:
        super().__init__(fpath, basename=basename)

    def save_file(self, file: torch.Tensor, **kwargs):
        buffer = io.BytesIO()
        torch.save(obj=file, f=buffer)
        self.zip_file.writestr(self.get_current_name() + ".pt", buffer.getvalue())


def make_archive(
    zip_name: str = "source.zip",
    source_dir: _PathLike = "./",
    dest_dir: _PathLike = "./",
    ignored_lsit: List = None,
    ignored_file: _PathLike = None,
) -> None:
    """function for saving a source code folder into a zipfile and possibly ignoring some directories

    Args:
        zip_name (str, optional): The name of the final zipfile. Defaults to 'source.zip'.
        source_dir (_PathLike, optional): Path to the source code directory. Defaults to "./".
        dest_dir (_PathLike, optional): Path to the directory where the zipfile will be saved. Defaults to "./".
        ignored_lsit (List, optional): List of directories to ignore. Defaults to None.
        ignored_file (_PathLike, optional): Path to a textfile containing the list of directories to ignore. In case that ignored_lsit is not None, this directories will be appended to ignored_lsit.  Defaults to None.

    Raises:
        IOError: if source directory does not exist
        IOError: if destination directory does not exist
    """
    source_dir = pathlib.Path(source_dir)
    dest_dir = pathlib.Path(dest_dir)

    if not source_dir.is_dir():
        raise IOError(f"{source_dir} must be a directory")
    if not dest_dir.is_dir():
        raise IOError(f"{dest_dir} must be a directory")

    should_ignore = [] if ignored_lsit is None else ignored_lsit
    if ignored_file is not None:
        with open(ignored_file, "r") as f:
            should_ignore.extend(f.read().splitlines())

    should_ignore = [os.path.relpath(path) for path in should_ignore]
    save_dir = dest_dir.joinpath(zip_name)

    with zipfile.ZipFile(save_dir, "w") as zf:
        for dirname, subdirs, files in os.walk(source_dir):
            for x in should_ignore:
                if x in subdirs:
                    subdirs.remove(x)
            zf.write(dirname, arcname=os.path.join("source", dirname))
            for filename in files:
                if filename == zip_name:
                    continue
                zf.write(
                    os.path.join(dirname, filename),
                    arcname=os.path.join("source", dirname, filename),
                )


def timing(f: Callable) -> None:
    """Python wrapper for measuring the runtime of a function

    Args:
        f (Callable): the input function
    """

    @functools.wraps(f)
    def wrap(*args, **kw):
        ts = time.process_time()
        result = f(*args, **kw)
        te = time.process_time()
        print(f"func:{f.__name__} took: {te-ts:2.4f} sec")
        return result

    return wrap


def extract_faces(images, face_bbox, face_size=64):
    """Given the face bounding boxes and a batch of images, crops each image based on its bbox and then resizes all outputs to face loss"""
    result = []
    for image, bbox in zip(images, face_bbox):
        x1, y1, x2, y2 = bbox.tolist()
        height, width = y2 - y1, x2 - x1
        if height <= 0 or width <= 0:
            result.append(torch.zeros(1, 3, face_size, face_size, device=image.device))
        else:
            image_ = image[:, y1:y2, x1:x2]
            result.append(
                torch.nn.functional.interpolate(
                    image_[None], size=face_size, mode="nearest"
                )
            )
    return torch.cat(result, dim=0)
