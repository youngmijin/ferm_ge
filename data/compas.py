import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import Dataset


class COMPAS(Dataset):
    """
    Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016).
    Machine bias. ProPublica, May, 23.
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_test: NDArray[np.float_] | None = None
        self.y_test: NDArray[np.float_] | None = None

        self.group_1_name = "African-American"
        self.group_1_train_indices: NDArray[np.intp] | None = None
        self.group_1_test_indices: NDArray[np.intp] | None = None

        self.group_2_name = "Caucasian"
        self.group_2_train_indices: NDArray[np.intp] | None = None
        self.group_2_test_indices: NDArray[np.intp] | None = None

    @property
    def name(self) -> str:
        return "compas"

    @property
    def file_local_path(self) -> str:
        return os.path.join(
            os.path.dirname(__file__), "compas_scores_two_years.csv"
        )

    @property
    def file_remote_url(self) -> str:
        return "https://b31.sharepoint.com/:x:/g/ETJqjlIrvvZGq98Knhmzc40Blx_z4fAC6AwiLExZbcRjgw?e=waxU8M&download=1"

    @property
    def file_md5_hash(self) -> str:
        return "9165d40c400bba93a8cffece2b74622b"

    def load(self):
        compas = pd.read_csv(self.file_local_path)
        compas = compas[compas["days_b_screening_arrest"] <= 30]
        compas = compas[compas["days_b_screening_arrest"] >= -30]
        compas = compas[compas["is_recid"] != -1]
        compas = compas[compas["c_charge_degree"] != "O"]
        compas = compas[compas["score_text"] != "N/A"]
        compas = compas[compas["race"].isin(["African-American", "Caucasian"])]
        compas = compas[
            [
                "age",
                "c_charge_degree",
                "race",
                "age_cat",
                "score_text",
                "sex",
                "priors_count",
                "days_b_screening_arrest",
                "decile_score",
                "is_recid",
                "two_year_recid",
            ]
        ]
        compas = compas.dropna()
        compas.replace(
            {"score_text": {"Low": 0, "Medium": 1, "High": 1}}, inplace=True
        )

        X = compas.drop(columns="score_text")
        y = compas["score_text"]

        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )  # type: ignore

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        self.group_1_train_indices = X_train.index[  # type: ignore
            X_train["race"] == "African-American"  # type: ignore
        ].to_numpy()
        self.group_2_train_indices = X_train.index[  # type: ignore
            X_train["race"] == "Caucasian"  # type: ignore
        ].to_numpy()

        self.group_1_test_indices = X_test.index[  # type: ignore
            X_test["race"] == "African-American"  # type: ignore
        ].to_numpy()
        self.group_2_test_indices = X_test.index[  # type: ignore
            X_test["race"] == "Caucasian"  # type: ignore
        ].to_numpy()

        categorical_features = X.select_dtypes(include=["object"]).columns
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

        self.X_train = preprocessor.fit_transform(X_train)  # type: ignore
        self.X_test = preprocessor.transform(X_test)  # type: ignore

        self.y_train = y_train.to_numpy()  # type: ignore
        self.y_test = y_test.to_numpy()  # type: ignore

    @property
    def train_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_train is not None, "X_train is not loaded"
        assert self.y_train is not None, "y_train is not loaded"
        return self.X_train, self.y_train

    @property
    def test_data(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        assert self.X_test is not None, "X_test is not loaded"
        assert self.y_test is not None, "y_test is not loaded"
        return self.X_test, self.y_test

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
    def test_group_indices(self) -> dict[str, NDArray[np.intp]]:
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
