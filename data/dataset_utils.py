from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rich import print

GroupCriteria = tuple[str, dict[str, list[int]]]


def make_group_indices(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, *criterias: GroupCriteria
) -> dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]]:
    group_names: list[list[str]] = []
    for _, group_mapping_dict in criterias:
        group_names.append(list(group_mapping_dict.keys()))
    group_name_combinations: list[list[str]] = [
        list(x) for x in product(*group_names)
    ]

    group_indices: dict[str, tuple[NDArray[np.intp], NDArray[np.intp]]] = {}
    for gnc in group_name_combinations:
        group_divisors_train: list = []
        group_divisors_valid: list = []
        for group_name, (group_key, group_mapping_dict) in zip(gnc, criterias):
            group_divisors_train.append(
                X_train[group_key].isin(group_mapping_dict[group_name])
            )
            group_divisors_valid.append(
                X_valid[group_key].isin(group_mapping_dict[group_name])
            )
        group_divisor_train = np.all(group_divisors_train, axis=0)
        group_divisor_valid = np.all(group_divisors_valid, axis=0)

        if sum(group_divisor_train) == 0:
            print(f"skipping group index {gnc} because it has no train data")
            continue
        if sum(group_divisor_valid) == 0:
            print(f"skipping group index {gnc} because it has no valid data")
            continue

        group_indices["-".join(gnc)] = (
            X_train.index[group_divisor_train].to_numpy(),
            X_valid.index[group_divisor_valid].to_numpy(),
        )

    return group_indices


def encode_onehot_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        for new_colname in df[column].unique():
            df[new_colname] = 0
            df.loc[df[column] == new_colname, new_colname] = 1
        df = df.drop(columns=column)
    return df
