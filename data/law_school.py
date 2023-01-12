import os

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
        self.X_test: NDArray[np.float_] | None = None
        self.y_test: NDArray[np.float_] | None = None

        self.group_1_name = "female"
        self.group_1_train_indices: NDArray[np.intp] | None = None
        self.group_1_test_indices: NDArray[np.intp] | None = None

        self.group_2_name = "male"
        self.group_2_train_indices: NDArray[np.intp] | None = None
        self.group_2_test_indices: NDArray[np.intp] | None = None

        self.load_files()

    @property
    def name(self) -> str:
        return "law_school"

    __file_local_path = os.path.join(
        os.path.dirname(__file__), "law_dataset.csv"
    )
    __file_remote_url = "https://b31.sharepoint.com/:x:/g/EaxpG0Gd-tZGiS6n4h11lkYBP3oAcTC3LieLiNmxU8pJdg?e=9vOEBj&download=1"

    def download(self):
        response = requests.get(self.__file_remote_url)
        assert response.status_code == 200, "Failed to download the dataset"
        with open(self.__file_local_path, "wb") as f:
            f.write(response.content)

    def check_files(self) -> bool:
        return os.path.isfile(self.__file_local_path)

    def load_files(self):
        if not self.check_files():
            self.download()

        law = pd.read_csv(self.__file_local_path)
        law = law.dropna()
        law = law.reset_index(drop=True)

        X = law.drop(columns=["pass_bar"])
        y = law["pass_bar"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.group_1_train_indices = X_train.index[  # type: ignore
            X_train["male"] == 0.0  # type: ignore
        ].to_numpy()
        self.group_2_train_indices = X_train.index[  # type: ignore
            X_train["male"] != 0.0  # type: ignore
        ].to_numpy()

        self.group_1_test_indices = X_test.index[  # type: ignore
            X_test["male"] == 0.0  # type: ignore
        ].to_numpy()
        self.group_2_test_indices = X_test.index[  # type: ignore
            X_test["male"] != 0.0  # type: ignore
        ].to_numpy()

        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
            ]
        )

        self.X_train = preprocessor.fit_transform(X_train)  # type: ignore
        self.X_test = preprocessor.transform(X_test)  # type: ignore

        self.y_train = y_train.to_numpy()  # type: ignore
        self.y_test = y_test.to_numpy()  # type: ignore

    def get_train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    def get_test_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_test is not None, "X_test is not loaded"
        assert self.y_test is not None, "y_test is not loaded"
        return self.X_test, self.y_test

    def get_train_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert (
            self.group_1_train_indices is not None
        ), "group_1_train_indices is not loaded"
        assert (
            self.group_2_train_indices is not None
        ), "group_2_train_indices is not loaded"
        return {
            self.group_1_name: self.group_1_train_indices,
            self.group_2_name: self.group_2_train_indices,
        }

    def get_test_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert (
            self.group_1_test_indices is not None
        ), "group_1_test_indices is not loaded"
        assert (
            self.group_2_test_indices is not None
        ), "group_2_test_indices is not loaded"
        return {
            self.group_1_name: self.group_1_test_indices,
            self.group_2_name: self.group_2_test_indices,
        }
