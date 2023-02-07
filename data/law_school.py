import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .dataset import Dataset


class LawSchool(Dataset):
    """
    Wightman, L. F. (1998).
    LSAC national longitudinal bar passage study.
    LSAC research report series.
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_indices: dict[
            str, tuple[NDArray[np.intp], NDArray[np.intp]]
        ] | None = None

    @property
    def name(self) -> str:
        return "law_school"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "law_dataset.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://www.dropbox.com/s/heyekivhlh0ipyo/law_dataset.csv?dl=1"

    @property
    def file_md5_hash(self) -> str:
        return "3296294f79ddd38d8f5fe31499f6ee12"

    def load(self, group_size: int = 2):
        law = pd.read_csv(self.file_local_path)
        law = law.dropna()
        law = law.reset_index(drop=True)

        X = law.drop(columns=["pass_bar"])
        y = law["pass_bar"]

        X_train: pd.DataFrame
        X_valid: pd.DataFrame
        y_train: pd.Series
        y_valid: pd.Series
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, random_state=42
        )  # type: ignore

        X_train = X_train.reset_index(drop=True)
        X_valid = X_valid.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_valid = y_valid.reset_index(drop=True)

        if group_size == 2:
            self.group_indices = {
                "female": (
                    X_train.index[X_train["male"] == 0.0].to_numpy(),
                    X_valid.index[X_valid["male"] == 0.0].to_numpy(),
                ),
                "male": (
                    X_train.index[X_train["male"] != 0.0].to_numpy(),
                    X_valid.index[X_valid["male"] != 0.0].to_numpy(),
                ),
            }
        else:
            raise ValueError("Invalid group size")

        self.X_train = X_train.to_numpy()
        self.X_valid = X_valid.to_numpy()

        self.y_train = y_train.to_numpy()
        self.y_valid = y_valid.to_numpy()

    @property
    def train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    @property
    def valid_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_valid is not None, "X_valid is not loaded"
        assert self.y_valid is not None, "y_valid is not loaded"
        return self.X_valid, self.y_valid

    @property
    def train_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert self.group_indices is not None, "group_indices is not loaded"
        return {k: v[0] for k, v in self.group_indices.items()}

    @property
    def valid_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert self.group_indices is not None, "group_indices is not loaded"
        return {k: v[1] for k, v in self.group_indices.items()}
