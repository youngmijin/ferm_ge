from abc import ABC, abstractmethod
from typing import List, Tuple, Type

import numpy as np

from ferm_ge.tasks import BaseTask


class Preset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def applicable_task(self) -> Type[BaseTask]:
        pass

    @abstractmethod
    def download(self):
        pass

    @abstractmethod
    def check_files(self) -> bool:
        pass

    @abstractmethod
    def load_files(self):
        pass

    @abstractmethod
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
