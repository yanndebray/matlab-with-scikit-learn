"""Compare a scikit-learn classifier and a MATLAB classifier via the MATLAB
Engine for Python, framed with skore.

Unlike ``callPython.m`` (where MATLAB drives and calls Python through py.*),
here Python drives and reaches into a *running* shared MATLAB session.

Setup
-----
1. In MATLAB, share the session so Python can attach to it:

       >> matlab.engine.shareEngine

2. Install the MATLAB Engine for Python (one-time, from a MATLAB install):

       cd "$MATLABROOT/extern/engines/python" && python -m pip install .

3. From Python:

       >>> import matlab.engine
       >>> m = matlab.engine.connect_matlab()
       >>> from compare_engine import run
       >>> run(m)
"""

from __future__ import annotations

import argparse

import matlab  # provided by the MATLAB Engine for Python
import matlab.engine
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skore import ComparisonReport, EstimatorReport

from compare import FrozenClassifier, get_split


# Each entry holds:
#   train       — the MATLAB call that fits ``mdl``.
#   predict     — a MATLAB snippet that produces ``yhat``, ``scores``, and
#                 ``class_names`` from ``mdl`` and ``X_test``.
# ``scores`` is an n_samples × n_classes posterior probability matrix; columns
# are ordered by ``class_names`` (which we use to align with sklearn's
# ``classes_`` on the Python side).
MATLAB_TRAINERS = {
    "fitcecoc": dict(
        train="mdl = fitcecoc(X_train, y_train, 'FitPosterior', true);",
        predict=(
            "[yhat, ~, ~, scores] = predict(mdl, X_test); "
            "yhat = double(yhat); class_names = double(mdl.ClassNames);"
        ),
    ),
    "fitctree": dict(
        train="mdl = fitctree(X_train, y_train);",
        predict=(
            "[yhat, scores] = predict(mdl, X_test); "
            "yhat = double(yhat); class_names = double(mdl.ClassNames);"
        ),
    ),
    "fitcknn(k=5)": dict(
        train="mdl = fitcknn(X_train, y_train, 'NumNeighbors', 5);",
        predict=(
            "[yhat, scores] = predict(mdl, X_test); "
            "yhat = double(yhat); class_names = double(mdl.ClassNames);"
        ),
    ),
    "TreeBagger(50)": dict(
        train="mdl = TreeBagger(50, X_train, y_train);",
        predict=(
            "[yhat, scores] = predict(mdl, X_test); "
            "yhat = str2double(yhat); class_names = str2double(mdl.ClassNames);"
        ),
    ),
}


class FrozenProbaClassifier(FrozenClassifier):
    """``FrozenClassifier`` that also serves precomputed class probabilities.

    Skore probes ``_can_skip_predict`` by sampling rows from ``X_test`` and
    calling ``predict`` / ``predict_proba`` on that subset; we therefore look
    cached predictions up by row content against ``X_ref`` so subsetting works.
    """

    def __init__(self, y_pred, y_proba, classes, X_ref):
        super().__init__(y_pred=y_pred, classes=classes)
        self.y_proba = np.asarray(y_proba, dtype=float)
        self.X_ref = np.asarray(X_ref)

    def _row_index(self, X):
        X = np.asarray(X)
        if X.shape == self.X_ref.shape and np.array_equal(X, self.X_ref):
            return slice(None)
        idx = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            matches = np.flatnonzero(np.all(self.X_ref == row, axis=1))
            if len(matches) == 0:
                raise ValueError(
                    "FrozenProbaClassifier received a row not present in X_ref"
                )
            idx[i] = matches[0]
        return idx

    def predict(self, X):
        return self.y_pred[self._row_index(X)]

    def predict_proba(self, X):
        return self.y_proba[self._row_index(X)]


def _matlab_predict(eng, X_train, y_train, X_test, classifier):
    """Train ``classifier`` in the engine and pull back labels + posteriors."""
    if classifier not in MATLAB_TRAINERS:
        raise ValueError(
            f"Unknown MATLAB classifier {classifier!r}. "
            f"Choose from {sorted(MATLAB_TRAINERS)}."
        )
    spec = MATLAB_TRAINERS[classifier]

    eng.workspace["X_train"] = matlab.double(np.asarray(X_train).tolist())
    eng.workspace["X_test"]  = matlab.double(np.asarray(X_test).tolist())
    eng.workspace["y_train"] = matlab.double(np.asarray(y_train).reshape(-1, 1).tolist())

    eng.eval(spec["train"], nargout=0)
    eng.eval(spec["predict"], nargout=0)

    y_pred = np.asarray(eng.workspace["yhat"]).ravel().astype(int)
    scores = np.asarray(eng.workspace["scores"], dtype=float)
    class_names = np.asarray(eng.workspace["class_names"]).ravel().astype(int)
    return y_pred, scores, class_names


def run(eng=None, dataset="iris", matlab_classifier="fitcecoc"):
    """Build a skore ComparisonReport between sklearn and MATLAB on one dataset.

    Parameters
    ----------
    eng : matlab.engine.MatlabEngine, optional
        A live engine handle. If omitted, attaches to any shared session via
        ``matlab.engine.connect_matlab()``.
    dataset : str
        One of the keys in ``compare.DATASETS`` (iris, wine, breast_cancer, digits).
    matlab_classifier : str
        Key in ``MATLAB_TRAINERS``.
    """
    if eng is None:
        print("Connecting to shared MATLAB session ...")
        eng = matlab.engine.connect_matlab()
        print(f"Connected to MATLAB (matlabroot = {eng.matlabroot()}).")

    X_train, X_test, y_train, y_test = get_split(dataset)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    classes = np.unique(np.concatenate([y_train, y_test]))

    sk_name = "LogisticRegression"
    sk_est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    sk_est.fit(X_train, y_train)
    sk_report = EstimatorReport(
        sk_est, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )

    print(f"Training MATLAB / {matlab_classifier} on {dataset} via engine ...")
    y_pred_ml, scores_ml, ml_class_names = _matlab_predict(
        eng, X_train, y_train, X_test, matlab_classifier
    )
    # Reorder MATLAB's score columns to match sklearn's classes_ ordering.
    col_idx = [int(np.where(ml_class_names == c)[0][0]) for c in classes]
    proba_ml = scores_ml[:, col_idx]
    ml_est = FrozenProbaClassifier(
        y_pred=y_pred_ml, y_proba=proba_ml, classes=classes, X_ref=X_test,
    ).fit(X_train, y_train)
    ml_report = EstimatorReport(
        ml_est, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    )

    comparison = ComparisonReport(reports={
        f"sklearn / {sk_name}":           sk_report,
        f"MATLAB / {matlab_classifier}":  ml_report,
    })

    print(f"\n=== skore ComparisonReport — {dataset} ===")
    print(comparison.metrics.summarize().frame())
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="iris",
                        help="iris | wine | breast_cancer | digits")
    parser.add_argument("--matlab-classifier", default="fitcecoc",
                        choices=sorted(MATLAB_TRAINERS))
    args = parser.parse_args()
    run(dataset=args.dataset, matlab_classifier=args.matlab_classifier)
