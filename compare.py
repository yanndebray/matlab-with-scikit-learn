"""Benchmark MATLAB and scikit-learn classifiers across multiple datasets with skore.

MATLAB drives the workflow: for each dataset it asks Python for the train/test
split, trains a fleet of MATLAB classifiers, and hands the predictions back.
Python then trains a matching fleet of sklearn classifiers, wraps each MATLAB
prediction set in a ``FrozenClassifier`` so skore can build an
``EstimatorReport`` for it, assembles a per-dataset ``ComparisonReport`` and
renders a multi-section HTML site that ships to GitHub Pages.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from skore import ComparisonReport, EstimatorReport


DATASETS = {
    "iris": dict(
        label="Iris",
        description="Multiclass — flower species (3 classes · 150 samples · 4 features)",
        loader=load_iris,
    ),
    "wine": dict(
        label="Wine",
        description="Multiclass — wine cultivars (3 classes · 178 samples · 13 features)",
        loader=load_wine,
    ),
    "breast_cancer": dict(
        label="Breast cancer",
        description="Binary — malignant vs benign (2 classes · 569 samples · 30 features)",
        loader=load_breast_cancer,
    ),
    "digits": dict(
        label="Digits",
        description="Multiclass — handwritten digits (10 classes · 1797 samples · 64 features)",
        loader=load_digits,
    ),
}


def sklearn_classifiers():
    """Fresh estimators each call so fit state never leaks across datasets."""
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "DecisionTree": DecisionTreeClassifier(random_state=0),
        "RandomForest(50)": RandomForestClassifier(n_estimators=50, random_state=0),
        "KNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


class FrozenClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible estimator that returns precomputed MATLAB predictions."""

    def __init__(self, y_pred, classes):
        self.y_pred = np.asarray(y_pred).ravel()
        self.classes_ = np.asarray(classes)

    def fit(self, X, y):
        return self

    def predict(self, X):
        if len(X) != len(self.y_pred):
            raise ValueError(
                f"FrozenClassifier has {len(self.y_pred)} cached predictions "
                f"but predict() received {len(X)} rows"
            )
        return self.y_pred

    def __sklearn_is_fitted__(self):
        return True


def dataset_keys():
    return list(DATASETS)


def get_split(name, test_size=0.25, random_state=0):
    info = DATASETS[name]
    X, y = info["loader"](return_X_y=True)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _as_int_array(arr):
    return np.asarray(arr).ravel().astype(int)


def _run_dataset(name, matlab_preds):
    info = DATASETS[name]
    X_train, X_test, y_train, y_test = get_split(name)
    y_train = _as_int_array(y_train)
    y_test = _as_int_array(y_test)
    classes = np.unique(np.concatenate([y_train, y_test]))

    rows = []
    reports = {}

    for clf_name, est in sklearn_classifiers().items():
        est.fit(X_train, y_train)
        y_pred = _as_int_array(est.predict(X_test))
        label = f"sklearn / {clf_name}"
        rows.append(
            dict(
                label=label,
                short=clf_name,
                side="sklearn",
                accuracy=accuracy_score(y_test, y_pred),
                f1=f1_score(y_test, y_pred, average="macro"),
                y_pred=y_pred,
                est=est,
            )
        )
        reports[label] = EstimatorReport(
            est, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    for clf_name, y_pred in matlab_preds.items():
        y_pred = _as_int_array(y_pred)
        frozen = FrozenClassifier(y_pred=y_pred, classes=classes).fit(X_train, y_train)
        label = f"MATLAB / {clf_name}"
        rows.append(
            dict(
                label=label,
                short=clf_name,
                side="MATLAB",
                accuracy=accuracy_score(y_test, y_pred),
                f1=f1_score(y_test, y_pred, average="macro"),
                y_pred=y_pred,
                est=frozen,
            )
        )
        reports[label] = EstimatorReport(
            frozen, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    # Build the skore comparison so the project lives up to its name, even
    # though we render the metrics ourselves below.
    _ = ComparisonReport(reports=reports)

    return dict(name=name, info=info, rows=rows, y_test=y_test)


def _confusion_matrix_image(out_path, dataset_label, sk_row, ml_row, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, row, side_label in zip(
        axes, [ml_row, sk_row], [f"MATLAB · {ml_row['short']}", f"sklearn · {sk_row['short']}"]
    ):
        cm = confusion_matrix(y_test, row["y_pred"])
        ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{side_label}\n{row['accuracy']:.2%} accuracy", fontsize=11)
    fig.suptitle(dataset_label, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _scoreboard_html(rows):
    overall_best = max(rows, key=lambda r: r["accuracy"])
    side_best = {
        side: max((r for r in rows if r["side"] == side), key=lambda r: r["accuracy"])
        for side in ("MATLAB", "sklearn")
    }

    # Stable ordering: MATLAB first (so the "what MATLAB can do" framing reads
    # left-to-right), then sklearn — both internally sorted by accuracy desc.
    sort_key = lambda r: (-r["accuracy"], r["short"])
    ordered = sorted(
        (r for r in rows if r["side"] == "MATLAB"), key=sort_key
    ) + sorted(
        (r for r in rows if r["side"] == "sklearn"), key=sort_key
    )

    cards = []
    for r in ordered:
        classes = ["card", f"side-{r['side'].lower()}"]
        if r["label"] == overall_best["label"]:
            classes.append("champion")
        elif r["label"] == side_best[r["side"]]["label"]:
            classes.append("winner")
        cards.append(
            f"""<div class="{' '.join(classes)}">
              <div class="card-side">{r['side']}</div>
              <div class="card-label">{r['short']}</div>
              <div class="card-value">{r['accuracy']:.2%}</div>
              <div class="card-sub">macro-F1 {r['f1']:.3f}</div>
            </div>"""
        )
    return "".join(cards)


def _verdict(rows):
    side_best = {
        side: max((r for r in rows if r["side"] == side), key=lambda r: r["accuracy"])
        for side in ("MATLAB", "sklearn")
    }
    ml, sk = side_best["MATLAB"], side_best["sklearn"]
    if abs(ml["accuracy"] - sk["accuracy"]) < 1e-9:
        return f"Tie — both sides peak at {ml['accuracy']:.2%}."
    if ml["accuracy"] > sk["accuracy"]:
        return (
            f"<strong>MATLAB / {ml['short']}</strong> wins by "
            f"{ml['accuracy'] - sk['accuracy']:.2%} over sklearn's best "
            f"({sk['short']})."
        )
    return (
        f"<strong>sklearn / {sk['short']}</strong> wins by "
        f"{sk['accuracy'] - ml['accuracy']:.2%} over MATLAB's best "
        f"({ml['short']})."
    )


def _section(dataset_section, output_dir):
    name = dataset_section["name"]
    info = dataset_section["info"]
    rows = dataset_section["rows"]
    y_test = dataset_section["y_test"]

    sk_best = max((r for r in rows if r["side"] == "sklearn"), key=lambda r: r["accuracy"])
    ml_best = max((r for r in rows if r["side"] == "MATLAB"), key=lambda r: r["accuracy"])

    img_name = f"cm_{name}.png"
    _confusion_matrix_image(output_dir / img_name, info["label"], sk_best, ml_best, y_test)

    return f"""
  <section class="dataset" id="{name}">
    <h2>{info['label']}</h2>
    <p class="lede">{info['description']}</p>
    <div class="scoreboard">{_scoreboard_html(rows)}</div>
    <p class="verdict">{_verdict(rows)}</p>
    <div class="panel">
      <img src="{img_name}" alt="Confusion matrices for the best classifiers on {info['label']}">
    </div>
  </section>"""


def _summary(sections):
    by_side = {"MATLAB": [], "sklearn": []}
    for sec in sections:
        for side in by_side:
            best = max(
                (r for r in sec["rows"] if r["side"] == side),
                key=lambda r: r["accuracy"],
            )
            by_side[side].append(best["accuracy"])
    matlab_avg = float(np.mean(by_side["MATLAB"]))
    sklearn_avg = float(np.mean(by_side["sklearn"]))
    wins = {"MATLAB": 0, "sklearn": 0, "tie": 0}
    for sec in sections:
        ml = max((r for r in sec["rows"] if r["side"] == "MATLAB"), key=lambda r: r["accuracy"])
        sk = max((r for r in sec["rows"] if r["side"] == "sklearn"), key=lambda r: r["accuracy"])
        if abs(ml["accuracy"] - sk["accuracy"]) < 1e-9:
            wins["tie"] += 1
        elif ml["accuracy"] > sk["accuracy"]:
            wins["MATLAB"] += 1
        else:
            wins["sklearn"] += 1
    return f"""
  <section class="summary">
    <div class="summary-grid">
      <div class="summary-card">
        <div class="card-label">MATLAB best — average</div>
        <div class="card-value">{matlab_avg:.2%}</div>
        <div class="card-sub">across {len(sections)} datasets</div>
      </div>
      <div class="summary-card">
        <div class="card-label">sklearn best — average</div>
        <div class="card-value">{sklearn_avg:.2%}</div>
        <div class="card-sub">across {len(sections)} datasets</div>
      </div>
      <div class="summary-card">
        <div class="card-label">Per-dataset wins</div>
        <div class="card-value">{wins['MATLAB']} – {wins['sklearn']}</div>
        <div class="card-sub">{wins['tie']} tie{'s' if wins['tie'] != 1 else ''}</div>
      </div>
    </div>
  </section>"""


def _toc(sections):
    items = "".join(
        f'<li><a href="#{s["name"]}">{s["info"]["label"]}</a></li>' for s in sections
    )
    return f'<nav class="toc"><ul>{items}</ul></nav>'


_STYLE = """
:root {
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
  --matlab: #0076a8;
  --sklearn: #f7931e;
  --shadow: 0 1px 3px rgba(15,20,40,0.06), 0 8px 24px rgba(15,20,40,0.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1320;
    --panel: #161b2c;
    --panel-alt: #1d2338;
    --ink: #e7eaf3;
    --muted: #99a1b8;
    --line: #262d44;
    --accent: #8b8cf7;
    --accent-soft: #1f2342;
    --gold: #f1c453;
    --matlab: #4ec0e6;
    --sklearn: #ffb45a;
    --shadow: 0 1px 3px rgba(0,0,0,0.4), 0 12px 28px rgba(0,0,0,0.35);
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 1080px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem;
  line-height: 1.6;
  color: var(--ink);
  background: var(--bg);
}
.header { display: flex; align-items: baseline; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 0.25rem; }
.header h1 { margin: 0; font-size: 1.95rem; letter-spacing: -0.01em; }
.badge {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
  background: var(--accent-soft); color: var(--accent);
  padding: 0.2rem 0.6rem; border-radius: 999px; font-weight: 600;
}
.intro { color: var(--muted); margin: 0.25rem 0 1.75rem; max-width: 70ch; }
nav.toc { margin: 0 0 2rem; }
nav.toc ul {
  list-style: none; display: flex; gap: 0.5rem; flex-wrap: wrap; padding: 0; margin: 0;
}
nav.toc a {
  display: inline-block; padding: 0.3rem 0.75rem; border-radius: 999px;
  background: var(--panel-alt); color: var(--ink); text-decoration: none;
  font-size: 0.85rem; border: 1px solid var(--line);
}
nav.toc a:hover { background: var(--accent-soft); color: var(--accent); }
section.dataset { margin: 2.75rem 0 0; }
section.dataset h2 { margin: 0 0 0.25rem; font-size: 1.4rem; letter-spacing: -0.005em; }
section.dataset .lede { color: var(--muted); margin: 0 0 1.25rem; font-size: 0.95rem; }
.scoreboard, .summary-grid {
  display: grid; gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.summary-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-bottom: 0.5rem; }
.card, .summary-card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 0.95rem 1.1rem; box-shadow: var(--shadow); position: relative; overflow: hidden;
}
.card.side-matlab { border-top: 3px solid var(--matlab); }
.card.side-sklearn { border-top: 3px solid var(--sklearn); }
.card.winner { border-color: var(--gold); }
.card.champion { border: 2px solid var(--gold); }
.card.winner::after, .card.champion::after {
  position: absolute; top: 0.6rem; right: 0.7rem;
  font-size: 0.62rem; font-weight: 800; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--gold);
}
.card.winner::after { content: "best on side"; }
.card.champion::after { content: "champion"; }
.card-side {
  font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted);
}
.card.side-matlab .card-side { color: var(--matlab); }
.card.side-sklearn .card-side { color: var(--sklearn); }
.card-label { font-weight: 600; font-size: 0.95rem; margin: 0.1rem 0 0.3rem; }
.card-value {
  font-size: 1.7rem; font-weight: 700; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; line-height: 1.1;
}
.card-sub { color: var(--muted); font-size: 0.78rem; margin-top: 0.2rem; }
.summary-card .card-label { font-weight: 500; color: var(--muted); font-size: 0.8rem;
  text-transform: uppercase; letter-spacing: 0.06em; }
.summary-card .card-value { font-size: 2.1rem; }
.verdict {
  margin: 1rem 0; padding: 0.7rem 1rem;
  background: var(--accent-soft); color: var(--accent);
  border-radius: 8px; font-size: 0.95rem;
}
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
  padding: 1rem; box-shadow: var(--shadow); margin-top: 0.75rem;
}
img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
footer {
  margin-top: 3.5rem; color: var(--muted); font-size: 0.85rem;
  border-top: 1px solid var(--line); padding-top: 1rem;
}
a { color: var(--accent); }
"""


def _render_site(sections, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    section_html = "".join(_section(s, output_dir) for s in sections)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    matlab_names = sorted({r["short"] for s in sections for r in s["rows"] if r["side"] == "MATLAB"})
    sklearn_names = sorted({r["short"] for s in sections for r in s["rows"] if r["side"] == "sklearn"})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MATLAB vs scikit-learn — skore benchmark</title>
  <style>{_STYLE}</style>
</head>
<body>
  <div class="header">
    <h1>MATLAB vs scikit-learn</h1>
    <span class="badge">skore benchmark</span>
  </div>
  <p class="intro">
    {len(sections)} datasets · {len(matlab_names)} MATLAB classifiers vs
    {len(sklearn_names)} scikit-learn classifiers, framed with
    <a href="https://skore.probabl.ai">skore</a>.
    MATLAB drives the pipeline (Statistics &amp; ML Toolbox, R2025b) and
    Python trains the sklearn fleet on the exact same train/test splits.
  </p>
  <p class="intro" style="margin-top:-1rem">
    <strong>MATLAB:</strong> {", ".join(f"<code>{n}</code>" for n in matlab_names)}
    &nbsp;·&nbsp;
    <strong>sklearn:</strong> {", ".join(f"<code>{n}</code>" for n in sklearn_names)}
  </p>

  {_summary(sections)}

  {_toc(sections)}

  {section_html}

  <footer>Built {timestamp} by GitHub Actions. Source on
    <a href="https://github.com/yanndebray/matlab-with-scikit-learn">GitHub</a>.</footer>
</body>
</html>
"""
    index = output_dir / "index.html"
    index.write_text(html)
    return index


def render_all(matlab_predictions, output_dir=None):
    """Build the full multi-dataset comparison site.

    Parameters
    ----------
    matlab_predictions : dict[str, dict[str, array-like]]
        Outer key: dataset name (must be in ``DATASETS``).
        Inner key: MATLAB classifier name.
        Inner value: predictions on the test set (1-D, length matches the
        test split returned by ``get_split``).
    output_dir : str | Path, optional
        Where to write the site. Defaults to ``site/`` next to this file.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "site"
    output_dir = Path(output_dir).resolve()

    sections = []
    for name in dataset_keys():
        if name not in matlab_predictions:
            print(f"[render_all] skipping {name}: no MATLAB predictions provided")
            continue
        # Materialise the per-classifier dict (it may arrive as a Python dict
        # forwarded straight from MATLAB containing numpy arrays).
        preds = {
            str(k): np.asarray(v) for k, v in dict(matlab_predictions[name]).items()
        }
        section = _run_dataset(name, preds)
        sections.append(section)
        best = max(section["rows"], key=lambda r: r["accuracy"])
        print(
            f"[render_all] {DATASETS[name]['label']}: "
            f"champion = {best['label']} @ {best['accuracy']:.4f}"
        )

    if not sections:
        raise RuntimeError("No datasets rendered — matlab_predictions was empty")

    index_path = _render_site(sections, output_dir)
    print(f"Wrote {index_path}")
    return str(index_path)


# Backwards-compatible single-dataset entry point — kept so a smoke test that
# only knows about Iris still works.
def make_split(test_size=0.25, random_state=0):
    return get_split("iris", test_size=test_size, random_state=random_state)


def compare(X_train, X_test, y_train, y_test, y_pred_matlab, output_dir=None):
    return render_all(
        {"iris": {"fitcecoc": y_pred_matlab}}, output_dir=output_dir
    )
