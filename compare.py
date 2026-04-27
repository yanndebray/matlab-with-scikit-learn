"""Compare a MATLAB classifier and an sklearn classifier with skore.

The MATLAB side trains its own classifier and hands us only the predictions
on the test set. We wrap those predictions in a small sklearn-compatible
estimator (``FrozenClassifier``) so skore can build an ``EstimatorReport``
for it and put it next to a real sklearn report inside a ``ComparisonReport``.
The function ``compare`` also writes a self-contained HTML site under the
given output directory, ready to be served on GitHub Pages.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
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


def _skore_metrics_table(comparison):
    """Best-effort: ask skore for a metrics DataFrame and HTML-ify it."""
    metrics = comparison.metrics
    for name in ("report_metrics", "summarize"):
        fn = getattr(metrics, name, None)
        if callable(fn):
            try:
                df = fn()
                return df.to_html(
                    classes="metrics", border=0, float_format="{:0.4f}".format
                )
            except Exception:
                continue
    return None


def _render_html(output_dir, y_test, model_predictions, skore_table_html):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(model_predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, model_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    classification_blocks = []
    for name, y_pred in model_predictions.items():
        report_text = classification_report(y_test, y_pred, digits=4)
        classification_blocks.append(f"<h3>{name}</h3><pre>{report_text}</pre>")

    metrics_section = (
        skore_table_html
        if skore_table_html
        else "<p><em>skore did not expose a tabular metrics summary in this version.</em></p>"
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MATLAB vs scikit-learn — skore comparison</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      max-width: 960px;
      margin: 2.5rem auto;
      padding: 0 1.25rem;
      line-height: 1.55;
      color: #222;
      background: #fafafa;
    }}
    h1 {{ border-bottom: 2px solid #e3e3e3; padding-bottom: 0.5rem; }}
    h2 {{ margin-top: 2.25rem; }}
    code, pre {{
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
      background: #f0f0f0;
      border-radius: 4px;
    }}
    pre {{ padding: 0.75rem 1rem; overflow-x: auto; }}
    table.metrics {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      border-radius: 6px;
      overflow: hidden;
    }}
    table.metrics th, table.metrics td {{
      padding: 0.55rem 0.85rem;
      border-bottom: 1px solid #eee;
      text-align: right;
    }}
    table.metrics th {{ background: #f5f7fa; font-weight: 600; }}
    table.metrics tr td:first-child, table.metrics tr th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    footer {{ margin-top: 3rem; color: #777; font-size: 0.9rem; }}
    a {{ color: #0a66c2; }}
  </style>
</head>
<body>
  <h1>MATLAB vs scikit-learn — <em>skore</em> comparison</h1>
  <p>
    Iris classification: <code>fitcecoc</code> (MATLAB R2025b, Statistics &amp; ML Toolbox)
    versus <code>LogisticRegression</code> (scikit-learn), placed side-by-side with
    <a href="https://skore.probabl.ai">skore</a>.
  </p>

  <h2>Metrics</h2>
  {metrics_section}

  <h2>Confusion matrices</h2>
  <img src="confusion_matrices.png" alt="Confusion matrices for each model">

  <h2>Classification reports</h2>
  {''.join(classification_blocks)}

  <footer>Built {timestamp} by GitHub Actions.</footer>
</body>
</html>
"""
    index = output_dir / "index.html"
    index.write_text(html)
    return index


def compare(
    X_train, X_test, y_train, y_test, y_pred_matlab, output_dir="site"
):
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train).ravel().astype(int)
    y_test = np.asarray(y_test).ravel().astype(int)
    y_pred_matlab = np.asarray(y_pred_matlab).ravel().astype(int)

    sk = LogisticRegression(max_iter=200).fit(X_train, y_train)
    y_pred_sklearn = sk.predict(X_test)

    sk_report = EstimatorReport(
        sk, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    ml = FrozenClassifier(y_pred=y_pred_matlab, classes=np.unique(y_train))
    ml_report = EstimatorReport(
        ml, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    comparison = ComparisonReport(
        reports={
            "sklearn LogisticRegression": sk_report,
            "MATLAB fitcecoc": ml_report,
        }
    )
    print(comparison.metrics)

    index_path = _render_html(
        output_dir=output_dir,
        y_test=y_test,
        model_predictions={
            "sklearn LogisticRegression": y_pred_sklearn,
            "MATLAB fitcecoc": y_pred_matlab,
        },
        skore_table_html=_skore_metrics_table(comparison),
    )
    print(f"Wrote {index_path}")
    return str(index_path)
