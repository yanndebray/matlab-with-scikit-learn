"""Compare a MATLAB classifier and an sklearn classifier with skore.

The MATLAB side trains its own classifier and hands us only the predictions
on the test set. We wrap those predictions in a small sklearn-compatible
estimator (``FrozenClassifier``) so skore can build an ``EstimatorReport``
for it and put it next to a real sklearn report inside a ``ComparisonReport``.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport


class FrozenClassifier(BaseEstimator, ClassifierMixin):
    """Returns precomputed predictions — the bridge for the MATLAB model."""

    def __init__(self, y_pred, classes):
        self.y_pred = np.asarray(y_pred).ravel()
        self.classes_ = np.asarray(classes)

    def fit(self, X, y):
        return self

    def predict(self, X):
        if len(X) != len(self.y_pred):
            raise ValueError(
                f"FrozenClassifier was given {len(self.y_pred)} cached "
                f"predictions but predict() received {len(X)} rows"
            )
        return self.y_pred

    def __sklearn_is_fitted__(self):
        return True


def make_split(test_size=0.25, random_state=0):
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def compare(X_train, X_test, y_train, y_test, y_pred_matlab):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).ravel().astype(int)
    y_test = np.asarray(y_test).ravel().astype(int)
    y_pred_matlab = np.asarray(y_pred_matlab).ravel().astype(int)

    sk = LogisticRegression(max_iter=200).fit(X_train, y_train)
    sk_report = EstimatorReport(
        sk,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    ml = FrozenClassifier(y_pred=y_pred_matlab, classes=np.unique(y_train))
    ml_report = EstimatorReport(
        ml,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    comparison = ComparisonReport(
        reports={
            "sklearn LogisticRegression": sk_report,
            "MATLAB fitcecoc": ml_report,
        }
    )
    metrics = comparison.metrics
    print(metrics)
    return metrics
