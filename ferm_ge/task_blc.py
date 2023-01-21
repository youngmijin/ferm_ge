import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

__all__ = ["BinaryLogisticClassification"]


class BinaryLogisticClassification:
    def __init__(self, **kwargs):
        self.classifier = LogisticRegression(**kwargs)

        self.train_proba: NDArray[np.float_] | None = None
        self.train_y: NDArray[np.float_] | None = None

        self.valid_proba: NDArray[np.float_] | None = None
        self.valid_y: NDArray[np.float_] | None = None

        self.train_group_indices: dict[str, NDArray[np.intp]] | None = None
        self.valid_group_indices: dict[str, NDArray[np.intp]] | None = None

    def train(self, X: NDArray[np.float_], y: NDArray[np.float_]):
        """
        Train the model on the given data.
        To get the predictions on the training data, use `predict_train()`.
        """

        self.classifier.fit(X, y)
        self.train_proba = self.classifier.predict_proba(X)
        self.train_y = y

    def valid(self, X: NDArray[np.float_], y: NDArray[np.float_]):
        """
        Validate the model on the given data.
        To get the predictions on the validation data, use `predict_valid()`.
        """

        assert self.classifier.coef_ is not None, "classifier is not fitted yet"

        self.valid_proba = self.classifier.predict_proba(X)
        self.valid_y = y

    def set_group(
        self,
        train_group_indices: dict[str, NDArray[np.intp]],
        valid_group_indices: dict[str, NDArray[np.intp]],
    ):
        """
        Set the group names and indices.
        """

        assert (
            train_group_indices.keys() == valid_group_indices.keys()
        ), "train and valid groups are not the same"

        self.train_group_indices = train_group_indices
        self.valid_group_indices = valid_group_indices

    @property
    def group_names(self) -> list[str]:
        assert self.train_group_indices is not None, "groups are not set yet"

        return list(self.train_group_indices.keys())

    def predict_train(
        self,
        threshold: float = 0.5,
        group: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.train_proba is not None and self.train_y is not None
        ), "classifier is not fitted yet"

        if group is not None:
            assert (
                self.train_group_indices is not None
            ), "train groups are not set yet"
            assert (
                group in self.train_group_indices
            ), f"train group {group} is not found"
            indices = self.train_group_indices[group]
            y_hat = (self.train_proba[indices, 1] >= threshold).astype(int)
            return y_hat, confusion_matrix(self.train_y[indices], y_hat).ravel()

        y_hat = (self.train_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.train_y, y_hat).ravel()

    def predict_valid(
        self,
        threshold: float = 0.5,
        group: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.valid_proba is not None and self.valid_y is not None
        ), "classifier is not validated yet"

        if group is not None:
            assert (
                self.valid_group_indices is not None
            ), "valid groups are not set yet"
            assert (
                group in self.valid_group_indices
            ), f"valid group {group} is not found"
            indices = self.valid_group_indices[group]
            y_hat = (self.valid_proba[indices, 1] >= threshold).astype(int)
            return y_hat, confusion_matrix(self.valid_y[indices], y_hat).ravel()

        y_hat = (self.valid_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.valid_y, y_hat).ravel()
