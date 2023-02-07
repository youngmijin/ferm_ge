import os
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .dataset import Dataset


class CommunitiesAndCrime(Dataset):
    """
    Redmond, M. (2017).
    U.S. Department of Commerce, Bureau of the Census, Census Of Population And Housing 1990 United States: Summary Tape File 1a & 3a (Computer Files), U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992), U.S. Department of Justice, Bureau of Justice Statistics, Law Enforcement Management And Administrative Statistics (Computer File) U.S. Department Of Commerce, Bureau Of The Census Producer, Washington, DC and Inter-university Consortium for Political and Social Research Ann Arbor, Michigan. (1992), U.S. Department of Justice, Federal Bureau of Investigation, Crime in the United States (Computer File) (1995), Redmond, M. A. and A. Baveja: A Data-Driven Software Tool for Enabling Cooperative Information Sharing Among Police Departments. European Journal of Operational Research 141 (2002) 660-678.
    """

    def __init__(self):
        warnings.warn(
            "The Communities and Crime dataset may cause overfitting. In some "
            "cases, the false positive rate can be zero.",
            UserWarning,
        )

        self.X_train: NDArray[np.float_] | None = None
        self.y_train: NDArray[np.float_] | None = None
        self.X_valid: NDArray[np.float_] | None = None
        self.y_valid: NDArray[np.float_] | None = None

        self.group_indices: dict[
            str, tuple[NDArray[np.intp], NDArray[np.intp]]
        ] | None = None

    @property
    def name(self) -> str:
        return "crime"

    @property
    def file_local_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "crimedata.csv")

    @property
    def file_remote_url(self) -> str:
        return "https://www.dropbox.com/s/2oscje2cdbzmlfx/crimedata.csv?dl=1"

    @property
    def file_md5_hash(self) -> str:
        return "3ff412ab144c7096c1c10c63c24c089a"

    def load(self, group_size: int = 2):
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
        crime = crime.reset_index(drop=True)

        crime["class"] = (
            crime["ViolentCrimesPerPop"]
            .astype(float)
            .apply(lambda x: 1 if x >= 1700 else 0)
        )
        crime = crime.drop(columns="ViolentCrimesPerPop")

        X = crime.drop(columns="class")
        y = crime["class"]

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
                "Lower-AfrAm-Rate": (
                    X_train.index[X_train["racepctblack"] < 30].to_numpy(),
                    X_valid.index[X_valid["racepctblack"] < 30].to_numpy(),
                ),
                "Higher-AfrAm-Rate": (
                    X_train.index[X_train["racepctblack"] >= 30].to_numpy(),
                    X_valid.index[X_valid["racepctblack"] >= 30].to_numpy(),
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
