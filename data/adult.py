import os

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import Dataset


class Adult(Dataset):
    """
    Kohavi, R. (1996).
    Scaling up the accuracy of Naive-Bayes classifiers: a decision-tree hybrid.
    In Proceedings of the 2nd International Conference on Knowledge Discovery
    and Data mining, Portland, 1996 (pp. 202–207).
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
        return "adult"

    __file_local_path = os.path.join(os.path.dirname(__file__), "adult.csv")
    __file_remote_url = "https://b31.sharepoint.com/:x:/g/ES-9TvClbj1ElsZOgMjiZQsBkFRNqLk0Sp1HUPPwC7yInA?e=xdCLvM&download=1"

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

        adult = pd.read_csv(self.__file_local_path)
        adult = adult.drop(columns=["fnlwgt"])
        adult = adult.replace({"?": np.nan})
        adult = adult.dropna()
        adult = adult.replace({"<=50K": 0, ">50K": 1})
        adult = adult.reset_index(drop=True)

        X = adult.drop(columns=["income"])
        y = adult["income"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.group_1_train_indices = X_train.index[  # type: ignore
            X_train["gender"] == "Female"  # type: ignore
        ].to_numpy()
        self.group_2_train_indices = X_train.index[  # type: ignore
            X_train["gender"] != "Female"  # type: ignore
        ].to_numpy()

        self.group_1_test_indices = X_test.index[  # type: ignore
            X_test["gender"] == "Female"  # type: ignore
        ].to_numpy()
        self.group_2_test_indices = X_test.index[  # type: ignore
            X_test["gender"] != "Female"  # type: ignore
        ].to_numpy()

        categorical_features = X.select_dtypes(
            include=["object", "bool"]
        ).columns
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_features),
                ("num", numerical_transformer, numerical_features),
            ]
        )

        self.X_train = preprocessor.fit_transform(X_train).A  # type: ignore
        self.X_test = preprocessor.transform(X_test).A  # type: ignore

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
