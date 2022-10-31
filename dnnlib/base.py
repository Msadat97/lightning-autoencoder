import abc
import pathlib
import zipfile
from typing import Any, Optional, Union


class BaseZipSaver(abc.ABC):
    def __init__(self, fpath: Union[str, pathlib.Path], basename: Optional[str] = "data") -> None:
        """the base interface for saving a batch of files (images for example) directly into a zipfile.

        Args:
            fpath (Union[str, pathlib.Path]): path to where the zipfile should be saved.
            basename (Optional[str], optional): the name of the base directory inside the zipfile. the files are saved in the form of basename/samplexxx. Defaults to "data".
        """

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
