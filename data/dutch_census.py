import os
from typing import Optional, Tuple, Type

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ferm_ge.tasks import BaseTask, BinaryLogisticClassificationTask

from .preset import Preset


class DutchCensus(Preset):
    def __init__(self):
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.load_files()

    @property
    def name(self) -> str:
        return "dutch_census"

    @property
    def applicable_task(self) -> Type[BaseTask]:
        return BinaryLogisticClassificationTask

    __file_local_path = os.path.join(
        os.path.dirname(__file__), "dutch_census_2001.csv"
    )
    __file_remote_url = "https://b31.sharepoint.com/:x:/g/EUw9H6-gTWJPiB7HFr7qmWgB5xx-XSXz92hwOOXs52PXig?e=vtDfb4&download=1"

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

        census = pd.read_csv(self.__file_local_path)
        census = census.dropna()
        census = census.replace({"5_4_9": 0, "2_1": 1})
        census = census.reset_index(drop=True)

        X = census.drop(columns=["occupation"])
        y = census["occupation"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

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

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.X_test is not None, "X_test is not loaded"
        assert self.y_test is not None, "y_test is not loaded"
        return self.X_test, self.y_test
