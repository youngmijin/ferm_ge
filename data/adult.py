import os

import numpy as np
import pandas as pd
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
    and Data mining, Portland, 1996 (pp. 202-207).
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_1_name = "female"
        self.group_1_train_indices: NDArray[np.intp] | None = None
        self.group_1_valid_indices: NDArray[np.intp] | None = None

        self.group_2_name = "male"
        self.group_2_train_indices: NDArray[np.intp] | None = None
        self.group_2_valid_indices: NDArray[np.intp] | None = None

    @property
    def name(self) -> str:
        return "adult"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "adult.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://b31.sharepoint.com/:x:/g/ES-9TvClbj1ElsZOgMjiZQsBkFRNqLk0Sp1HUPPwC7yInA?e=xdCLvM&download=1"

    @property
    def file_md5_hash(self) -> str:
        return "46a9b0988c83b02d27640bf9ced3ab95"

    def load(self):
        adult = pd.read_csv(self.file_local_path)
        adult = adult.drop(columns=["fnlwgt"])
        adult = adult.replace({"?": np.nan})
        adult = adult.dropna()
        adult = adult.replace({"<=50K": 0, ">50K": 1})
        adult = adult.reset_index(drop=True)

        X = adult.drop(columns=["income"])
        y = adult["income"]

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

        self.group_1_train_indices = X_train.index[  # type: ignore
            X_train["gender"] == "Female"  # type: ignore
        ].to_numpy()
        self.group_2_train_indices = X_train.index[  # type: ignore
            X_train["gender"] != "Female"  # type: ignore
        ].to_numpy()

        self.group_1_valid_indices = X_valid.index[  # type: ignore
            X_valid["gender"] == "Female"  # type: ignore
        ].to_numpy()
        self.group_2_valid_indices = X_valid.index[  # type: ignore
            X_valid["gender"] != "Female"  # type: ignore
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
        self.X_valid = preprocessor.transform(X_valid).A  # type: ignore

        self.y_train = y_train.to_numpy()  # type: ignore
        self.y_valid = y_valid.to_numpy()  # type: ignore

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

    @property
    def valid_group_indices(self) -> dict[str, NDArray[np.intp]]:
        assert (
            self.group_1_valid_indices is not None
        ), "group_1_valid_indices is not loaded"
        assert (
            self.group_2_valid_indices is not None
        ), "group_2_valid_indices is not loaded"
        return {
            self.group_1_name: self.group_1_valid_indices,
            self.group_2_name: self.group_2_valid_indices,
        }
