from dataclasses import dataclass
from typing import Optional
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


@dataclass
class TrainTestSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class BreastCancerDataLoader:
    def __init__(self,
                 feature_names: Optional[list] = None,
                 normalize: bool = False,
                 test_size: float = 0.2):
        self.feature_names = feature_names
        self.normalize = normalize
        self.test_size = test_size
        self.scalar_mean = None
        self.scalar_std = None

    def load_data(self):
        data = load_breast_cancer()
        all_feature_names = data.feature_names
        X = data.data
        y = data.target  # Already binary: 0 = malignant, 1 = benign

        if self.feature_names is not None:
            name_to_index = {name: i for i, name in enumerate(all_feature_names)}
            indices = [name_to_index[name] for name in self.feature_names]
            X = X[:, indices]
            if len(indices) == 1:
                X = X.flatten()

        return X, y

    def load_and_split_data(self) -> TrainTestSplit:
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        if self.normalize:
            if X_train.ndim == 1:
                self.scalar_mean, self.scalar_std = X_train.mean(), X_train.std()
            else:
                self.scalar_mean = X_train.mean(axis=0)
                self.scalar_std = X_train.std(axis=0)
            X_train = (X_train - self.scalar_mean) / self.scalar_std
            X_test = (X_test - self.scalar_mean) / self.scalar_std

        return TrainTestSplit(X_train, X_test, y_train, y_test)
