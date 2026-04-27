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
    accuracy_score,
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
    fig.savefig(
        output_dir / "confusion_matrices.png",
        dpi=140,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    classification_blocks = []
    accuracies = {}
    for name, y_pred in model_predictions.items():
        accuracies[name] = accuracy_score(y_test, y_pred)
        report_text = classification_report(y_test, y_pred, digits=4)
        classification_blocks.append(f"<h3>{name}</h3><pre>{report_text}</pre>")

    best = max(accuracies, key=accuracies.get)
    score_cards = "".join(
        f"""<div class="card{' winner' if name == best else ''}">
              <div class="card-label">{name}</div>
              <div class="card-value">{acc:.2%}</div>
              <div class="card-sub">accuracy on {len(y_test)} test samples</div>
            </div>"""
        for name, acc in accuracies.items()
    )
    if len(set(accuracies.values())) == 1:
        verdict = "Both classifiers agree on every test sample."
    else:
        delta = accuracies[best] - min(accuracies.values())
        verdict = f"<strong>{best}</strong> leads by {delta:.2%}."

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
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MATLAB vs scikit-learn — skore comparison</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-alt: #f3f5f9;
      --ink: #1c2230;
      --muted: #5b6478;
      --line: #e6e9ef;
      --accent: #4f46e5;
      --accent-soft: #eef0ff;
      --gold: #d4a017;
      --shadow: 0 1px 3px rgba(15,20,40,0.06), 0 8px 24px rgba(15,20,40,0.04);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0f1320;
        --panel: #161b2c;
        --panel-alt: #1d2338;
        --ink: #e7eaf3;
        --muted: #99a1b8;
        --line: #262d44;
        --accent: #8b8cf7;
        --accent-soft: #1f2342;
        --gold: #f1c453;
        --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 12px 28px rgba(0,0,0,0.35);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      max-width: 1040px;
      margin: 0 auto;
      padding: 2.5rem 1.5rem 4rem;
      line-height: 1.6;
      color: var(--ink);
      background: var(--bg);
    }}
    .header {{
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-bottom: 0.5rem;
    }}
    .header h1 {{ margin: 0; font-size: 1.85rem; letter-spacing: -0.01em; }}
    .badge {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: var(--accent-soft);
      color: var(--accent);
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-weight: 600;
    }}
    .lede {{ color: var(--muted); margin: 0.25rem 0 2rem; max-width: 60ch; }}
    h2 {{
      margin: 2.5rem 0 1rem;
      font-size: 1.15rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    h3 {{ margin: 1.5rem 0 0.5rem; font-size: 1rem; }}
    code, pre {{
      font-family: "SF Mono", "Menlo", "Consolas", monospace;
      background: var(--panel-alt);
      border-radius: 4px;
    }}
    code {{ padding: 0.05rem 0.35rem; font-size: 0.9em; }}
    pre {{
      padding: 0.9rem 1.1rem;
      overflow-x: auto;
      border: 1px solid var(--line);
      font-size: 0.85rem;
      line-height: 1.45;
    }}
    .scoreboard {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin: 0 0 0.5rem;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1.1rem 1.25rem;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }}
    .card.winner {{ border-color: var(--gold); }}
    .card.winner::after {{
      content: "best";
      position: absolute;
      top: 0.75rem;
      right: 0.75rem;
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--gold);
    }}
    .card-label {{ color: var(--muted); font-size: 0.85rem; font-weight: 500; }}
    .card-value {{
      font-size: 2.1rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0.35rem 0 0.1rem;
      font-variant-numeric: tabular-nums;
    }}
    .card-sub {{ color: var(--muted); font-size: 0.8rem; }}
    .verdict {{
      margin: 1rem 0 0;
      padding: 0.75rem 1rem;
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 8px;
      font-size: 0.95rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1.25rem;
      box-shadow: var(--shadow);
    }}
    table.metrics {{
      border-collapse: collapse;
      width: 100%;
      font-variant-numeric: tabular-nums;
    }}
    table.metrics th, table.metrics td {{
      padding: 0.55rem 0.85rem;
      border-bottom: 1px solid var(--line);
      text-align: right;
    }}
    table.metrics th {{ font-weight: 600; color: var(--muted); }}
    table.metrics tr:last-child td {{ border-bottom: none; }}
    table.metrics tr td:first-child, table.metrics tr th:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; border-radius: 8px; }}
    footer {{
      margin-top: 3.5rem;
      color: var(--muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--line);
      padding-top: 1rem;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="header">
    <h1>MATLAB vs scikit-learn</h1>
    <span class="badge">skore comparison</span>
  </div>
  <p class="lede">
    Iris classification — <code>fitcecoc</code> (MATLAB R2025b, Statistics &amp; ML Toolbox)
    versus <code>LogisticRegression</code> (scikit-learn), placed side-by-side with
    <a href="https://skore.probabl.ai">skore</a>.
  </p>

  <div class="scoreboard">{score_cards}</div>
  <p class="verdict">{verdict}</p>

  <h2>Metrics</h2>
  <div class="panel">{metrics_section}</div>

  <h2>Confusion matrices</h2>
  <div class="panel"><img src="confusion_matrices.png" alt="Confusion matrices for each model"></div>

  <h2>Classification reports</h2>
  <div class="panel">{''.join(classification_blocks)}</div>

  <footer>Built {timestamp} by GitHub Actions.</footer>
</body>
</html>
"""
    index = output_dir / "index.html"
    index.write_text(html)
    return index


def compare(
    X_train, X_test, y_train, y_test, y_pred_matlab, output_dir=None
):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "site"
    output_dir = Path(output_dir).resolve()

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
