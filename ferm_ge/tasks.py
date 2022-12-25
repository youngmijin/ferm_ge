from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class BaseTask(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model on the given data.
        To get the predictions on the training data, use `predict_train()` or
        `predict_train_with_threshold()`.
        """
        pass

    @abstractmethod
    def test(self, X: np.ndarray, y: np.ndarray):
        """
        Test the model on the given data.
        To get the predictions on the test data, use `predict_test()` or
        `predict_test_with_threshold()`.
        """
        pass

    @abstractmethod
    def predict_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict on the training data.
        """
        pass

    @abstractmethod
    def predict_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict on the test data.
        """
        pass

    @abstractmethod
    def predict_train_with_threshold(
        self, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on the training data with a given threshold.
        """
        pass

    @abstractmethod
    def predict_test_with_threshold(
        self, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on the test data with a given threshold.
        """
        pass

    @property
    @abstractmethod
    def trained(self) -> bool:
        """
        Whether the model is trained.
        """
        pass

    @property
    @abstractmethod
    def tested(self) -> bool:
        """
        Whether the model is tested.
        """
        pass


class BinaryLogisticClassificationTask(BaseTask):
    def __init__(self):
        self.classifier = LogisticRegression(max_iter=1000)

        self.is_trained = False
        self.train_pred: Optional[np.ndarray] = None
        self.train_confusion_matrix: Optional[np.ndarray] = None
        self.train_proba: Optional[np.ndarray] = None
        self.train_y: Optional[np.ndarray] = None

        self.is_tested = False
        self.test_pred: Optional[np.ndarray] = None
        self.test_confusion_matrix: Optional[np.ndarray] = None
        self.test_proba: Optional[np.ndarray] = None
        self.test_y: Optional[np.ndarray] = None

    def train(self, X: np.ndarray, y: np.ndarray):
        self.classifier.fit(X, y)

        self.train_pred = self.classifier.predict(X)
        self.train_confusion_matrix = confusion_matrix(y, self.train_pred)
        self.train_proba = self.classifier.predict_proba(X)
        self.train_y = y

        self.is_trained = True

    def test(self, X: np.ndarray, y: np.ndarray):
        assert self.is_trained, "Classifier is not fitted yet."

        self.test_pred = self.classifier.predict(X)
        self.test_confusion_matrix = confusion_matrix(y, self.test_pred)
        self.test_proba = self.classifier.predict_proba(X)
        self.test_y = y

        self.is_tested = True

    def predict_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.is_trained, "Classifier is not fitted yet."
        assert (
            self.train_pred is not None
            and self.train_proba is not None
            and self.train_confusion_matrix is not None
        ), "Classifier is not fitted yet."
        return (
            self.train_pred,
            self.train_proba,
            self.train_confusion_matrix.ravel(),
        )

    def predict_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.is_tested, "Classifier is not tested yet."
        assert (
            self.test_pred is not None
            and self.test_proba is not None
            and self.test_confusion_matrix is not None
        ), "Classifier is not tested yet."
        return (
            self.test_pred,
            self.test_proba,
            self.test_confusion_matrix.ravel(),
        )

    def predict_train_with_threshold(
        self, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.is_trained, "Classifier is not fitted yet."
        assert self.train_proba is not None, "Classifier is not fitted yet."
        y_hat = (self.train_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.train_y, y_hat).ravel()

    def predict_test_with_threshold(
        self, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.is_tested, "Classifier is not tested yet."
        assert self.test_proba is not None, "Classifier is not tested yet."
        y_hat = (self.test_proba[:, 1] >= threshold).astype(int)
        return y_hat, confusion_matrix(self.test_y, y_hat).ravel()

    @property
    def trained(self) -> bool:
        return self.is_trained

    @property
    def tested(self) -> bool:
        return self.is_tested
