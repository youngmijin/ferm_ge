import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import Dataset


class CommunitiesAndCrime(Dataset):
    """
    Redmond, M. (2017).
    U.S. Department of Commerce, Bureau of the Census, Census Of Population And Housing 1990 United States: Summary Tape File 1a & 3a (Computer Files), U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992), U.S. Department of Justice, Bureau of Justice Statistics, Law Enforcement Management And Administrative Statistics (Computer File) U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992), U.S. Department of Justice, Federal Bureau of Investigation, Crime in the United States (Computer File) (1995), Redmond, M. A. and A. Baveja: A Data-Driven Software Tool for Enabling Cooperative Information Sharing Among Police Departments. European Journal of Operational Research 141 (2002) 660-678.
    """

    def __init__(self):
        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_1_name = "African-American"
        self.group_1_train_indices: NDArray[np.intp] | None = None
        self.group_1_valid_indices: NDArray[np.intp] | None = None

        self.group_2_name = "Non-African-American"
        self.group_2_train_indices: NDArray[np.intp] | None = None
        self.group_2_valid_indices: NDArray[np.intp] | None = None

    @property
    def name(self) -> str:
        return "crime"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "crimedata.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://b31.sharepoint.com/:x:/g/EYVchXteXn9Gg9QUb2QQbOMBpsT-FZA6CKvuNKK8YRy90A?e=mftpOt&download=1"

    @property
    def file_md5_hash(self) -> str:
        return "3ff412ab144c7096c1c10c63c24c089a"

    def load(self):
        crime = pd.read_csv(self.file_local_path)
        crime = crime[
            [
                "racepctblack",
                "pctWInvInc",
                "pctWPubAsst",
                "NumUnderPov",
                "PctPopUnderPov",
                "PctUnemployed",
                "MalePctDivorce",
                "FemalePctDiv",
                "TotalPctDiv",
                "PersPerFam",
                "PctKids2Par",
                "PctYoungKids2Par",
                "PctTeen2Par",
                "PctPersOwnOccup",
                "HousVacant",
                "PctHousOwnOcc",
                "PctVacantBoarded",
                "NumInShelters",
                "NumStreet",
                "ViolentCrimesPerPop",
            ]
        ]
        crime = crime.replace({"?": np.nan})
        crime = crime.dropna()

        crime["ViolentCrimesPerPop"] = crime["ViolentCrimesPerPop"].astype(
            float
        )
        crime_ViolentCrimesPerPop_max = crime["ViolentCrimesPerPop"].max()
        crime_ViolentCrimesPerPop_min = crime["ViolentCrimesPerPop"].min()
        crime["ViolentCrimesPerPop"] = (
            crime["ViolentCrimesPerPop"] - crime_ViolentCrimesPerPop_min
        ) / (crime_ViolentCrimesPerPop_max - crime_ViolentCrimesPerPop_min)

        crime_ViolentCrimesPerPop_0_07_upper_loc = crime[
            crime["ViolentCrimesPerPop"] >= 0.07
        ].index
        crime_ViolentCrimesPerPop_0_07_lower_loc = crime[
            crime["ViolentCrimesPerPop"] < 0.07
        ].index
        crime.loc[
            crime_ViolentCrimesPerPop_0_07_upper_loc, "ViolentCrimesPerPop"
        ] = 1
        crime.loc[
            crime_ViolentCrimesPerPop_0_07_lower_loc, "ViolentCrimesPerPop"
        ] = 0

        X = crime.drop(columns="ViolentCrimesPerPop")
        y = crime["ViolentCrimesPerPop"]

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
        self.X_valid = preprocessor.transform(X_valid)  # type: ignore

        self.group_1_train_indices = X_train.index[  # type: ignore
            X_train["racepctblack"] >= 3  # type: ignore
        ].to_numpy()
        self.group_2_train_indices = X_train.index[  # type: ignore
            X_train["racepctblack"] < 3  # type: ignore
        ].to_numpy()

        self.group_1_valid_indices = X_valid.index[  # type: ignore
            X_valid["racepctblack"] >= 3  # type: ignore
        ].to_numpy()
        self.group_2_valid_indices = X_valid.index[  # type: ignore
            X_valid["racepctblack"] < 3  # type: ignore
        ].to_numpy()

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
