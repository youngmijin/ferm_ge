import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from .dataset import Dataset
from .dataset_utils import (
    GroupCriteria,
    encode_onehot_columns,
    make_group_indices,
)


class COMPAS(Dataset):
    """
    Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016).
    Machine bias. ProPublica, May, 23.
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
        return "compas"

    @property
    def file_local_path(self) -> str:
        return os.path.join(
            os.path.dirname(__file__), "compas_scores_two_years.csv"
        )

    @property
    def file_remote_url(self) -> str:
        return "https://www.dropbox.com/s/94lbdcu901akn4d/compas_scores_two_years.csv?dl=1"

    @property
    def file_md5_hash(self) -> str:
        return "9165d40c400bba93a8cffece2b74622b"

    def load(self, group_size: int = 2):
        compas = pd.read_csv(self.file_local_path)
        compas = compas[compas["days_b_screening_arrest"] <= 30]
        compas = compas[compas["days_b_screening_arrest"] >= -30]
        compas = compas[compas["is_recid"] != -1]
        compas = compas[compas["c_charge_degree"] != "O"]
        compas = compas[compas["score_text"] != "N/A"]
        compas = compas[compas["race"].isin(["African-American", "Caucasian"])]
        compas = compas[
            [
                "sex",
                "age",
                "age_cat",
                "race",
                "juv_fel_count",
                "juv_misd_count",
                "juv_other_count",
                "priors_count",
                "c_charge_degree",
                "score_text",
                "v_score_text",
                "two_year_recid",
            ]
        ]
        compas = compas.dropna()
        compas["sex"] = compas["sex"].replace({"Female": 1, "Male": 0})
        compas["race"] = compas["race"].replace(
            {"African-American": 0, "Caucasian": 1}
        )
        compas["score_text"] = compas["score_text"].replace(
            {"Low": 0, "Medium": 0, "High": 1}
        )
        compas["v_score_text"] = compas["v_score_text"].replace(
            {"Low": 0, "Medium": 0, "High": 1}
        )
        compas = encode_onehot_columns(compas, ["age_cat", "c_charge_degree"])
        compas["two_year_recid"] = (compas["two_year_recid"] - 1) * -1

        X = compas.drop(columns="two_year_recid")
        y = compas["two_year_recid"]

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

        race_gc: GroupCriteria = (
            "race",
            {"African-American": [0], "Caucasian": [1]},
        )

        if group_size == 2:
            self.group_indices = make_group_indices(X_train, X_valid, race_gc)
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
