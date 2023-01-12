from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class BinaryLogisticClassification:
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)

        self.train_proba: NDArray[np.float_] | None = None
        self.train_y: NDArray[np.float_] | None = None

        self.test_proba: NDArray[np.float_] | None = None
        self.test_y: NDArray[np.float_] | None = None

        self.train_group_indices: dict[str, NDArray[np.intp]] | None = None
        self.test_group_indices: dict[str, NDArray[np.intp]] | None = None

    def train(self, X: NDArray[np.float_], y: NDArray[np.float_]):
        """
        Train the model on the given data.
        To get the predictions on the training data, use `predict_train()`.
        """

        self.classifier.fit(X, y)
        self.train_proba = self.classifier.predict_proba(X)
        self.train_y = y

    def test(self, X: NDArray[np.float_], y: NDArray[np.float_]):
        """
        Test the model on the given data.
        To get the predictions on the test data, use `predict_test()`.
        """

        assert (
            self.classifier.coef_ is not None
        ), "Classifier is not fitted yet."

        self.test_proba = self.classifier.predict_proba(X)
        self.test_y = y

    def set_group(
        self,
        train_group_indices: dict[str, NDArray[np.intp]],
        test_group_indices: dict[str, NDArray[np.intp]],
    ):
        """
        Set the group names and indices.
        """

        self.train_group_indices = train_group_indices
        self.test_group_indices = test_group_indices

    def iter_train_groups(self) -> Iterator[tuple[str, NDArray[np.intp]]]:
        assert (
            self.train_group_indices is not None
        ), "group indices are not set yet."

        for group, indices in self.train_group_indices.items():
            yield group, indices

    def iter_test_groups(self) -> Iterator[tuple[str, NDArray[np.intp]]]:
        assert (
            self.test_group_indices is not None
        ), "group indices are not set yet."

        for group, indices in self.test_group_indices.items():
            yield group, indices

    @property
    def n_groups(self) -> int:
        assert (
            self.train_group_indices is not None
            and self.test_group_indices is not None
        ), "group indices are not set yet."
        assert len(self.train_group_indices) == len(
            self.test_group_indices
        ), "train and test groups are not the same."

        return len(self.train_group_indices)

    def predict_train(
        self,
        threshold: float = 0.5,
        indices: NDArray[np.intp] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.train_proba is not None and self.train_y is not None
        ), "Classifier is not fitted yet."

        if indices is not None:
            y_hat = (self.train_proba[indices, 1] >= threshold).astype(int)
            return y_hat, confusion_matrix(self.train_y[indices], y_hat).ravel()

        y_hat = (self.train_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.train_y, y_hat).ravel()

    def predict_test(
        self,
        threshold: float = 0.5,
        indices: NDArray[np.intp] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert (
            self.test_proba is not None and self.test_y is not None
        ), "Classifier is not tested yet."

        if indices is not None:
            y_hat = (self.test_proba[indices, 1] >= threshold).astype(int)
            return y_hat, confusion_matrix(self.test_y[indices], y_hat).ravel()

        y_hat = (self.test_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.test_y, y_hat).ravel()
