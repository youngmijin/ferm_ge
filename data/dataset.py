from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        pass

    @abstractmethod
    def get_test_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        pass

    @abstractmethod
    def get_train_group_indices(self) -> dict[str, NDArray[np.intp]]:
        pass

    @abstractmethod
    def get_test_group_indices(self) -> dict[str, NDArray[np.intp]]:
        pass
