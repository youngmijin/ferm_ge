import hashlib
import os
from abc import ABC, abstractmethod

import numpy as np
import requests
from numpy.typing import NDArray


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def file_local_path(self) -> str:
        pass

    @property
    @abstractmethod
    def file_remote_url(self) -> str:
        pass

    @property
    @abstractmethod
    def file_md5_hash(self) -> str:
        pass

    def check(self) -> bool:
        if os.path.isfile(self.file_local_path):
            with open(self.file_local_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest() == self.file_md5_hash
        return False

    def download(self, remote_url: str | None = None, force: bool = False):
        if self.check() and not force:
            return
        if remote_url is None:
            remote_url = self.file_remote_url
        response = requests.get(remote_url)
        assert response.status_code == 200, "failed to download the dataset"
        with open(self.file_local_path, "wb") as f:
            f.write(response.content)
        if not self.check():
            raise RuntimeError("downloaded file is corrupted")

    @abstractmethod
    def load(self, group_size: int):
        pass

    @property
    @abstractmethod
    def train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        pass

    @property
    @abstractmethod
    def valid_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        pass

    @property
    @abstractmethod
    def train_group_indices(self) -> dict[str, NDArray[np.intp]]:
        pass

    @property
    @abstractmethod
    def valid_group_indices(self) -> dict[str, NDArray[np.intp]]:
        pass
