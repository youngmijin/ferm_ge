import os
from typing import Optional, Tuple, Type

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ferm_ge.tasks import BaseTask, BinaryLogisticClassificationTask

from .preset import Preset


class Adult(Preset):
    def __init__(self):
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "adult"

    @property
    def applicable_task(self) -> Type[BaseTask]:
        return BinaryLogisticClassificationTask

    __file_local_path = os.path.join(os.path.dirname(__file__), "adult.csv")
    __file_remote_url = (
        "https://www.dropbox.com/s/h0nlmmcxe5n1dde/adult.csv?dl=1"
    )

    def download(self):
        response = requests.get(self.__file_remote_url)
        assert response.status_code == 200, "Failed to download the dataset"
        with open(self.__file_local_path, "wb") as f:
            f.write(response.content)

    def check_files(self) -> bool:
        return os.path.isfile(self.__file_local_path)

    def load_files(self):
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

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X_test is not None, "X_test is not loaded"
        assert self.y_test is not None, "y_test is not loaded"
        return self.X_test, self.y_test
