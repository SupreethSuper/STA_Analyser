"""Check whether the decision tree and random forest do classification or regression.

Classification predicts a category (here C1-C5). Regression predicts a number
(here a delay in ns). This script answers which one each model does, using four
separate kinds of evidence:

  1. SOURCE    Which scikit-learn models and metrics the two scripts use.
  2. MODELS    The fitted models from a real run of each script: what they were
               trained on, what they are, and what predict() returns.
  3. FILES     The output files each script wrote during that run.
  4. CONTROLS  Real regressors put through the same inspection, to prove it can
               report REGRESSION and does not simply always say CLASSIFICATION.

Nothing in the project is modified. The two scripts run from their own unedited
code, but this process points their results and feature cache at a temporary
folder that is deleted afterwards, so results/ and TimingPredict/cache are left
alone.

The run uses the 7 small --quick designs (5 train, 2 test) instead of all 21,
because the model type is fixed in the code and does not depend on the data. Any
scores produced along the way are therefore not the project's real scores.

Run with the TimingPredict environment, which has the DGL package needed to read
the graphs:

    ..\\TimingPredict\\.venv312\\Scripts\\python.exe check_tree_forest_output.py

Exit code 0 means every check passed, and 1 means at least one failed.
"""

import ast
import contextlib
import csv
import io
import json
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Set before the project modules load, so no __pycache__ is added beside them.
sys.dont_write_bytecode = True

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.multiclass import type_of_target

import decision_tree as dt
import paths
import random_forest_baseline as rf


HERE = Path(__file__).resolve().parent
CLASS_NAMES = ["C1", "C2", "C3", "C4", "C5"]
TASKS = (("net", "net_delay"), ("cell", "cell_delay"))
FOREST_JOBS = "4"
SEED = 42
REGRESSION_METRICS = {
    "r2_score", "mean_squared_error", "root_mean_squared_error",
    "mean_absolute_error", "mean_absolute_percentage_error",
}
REGRESSION_COLUMN = re.compile(r"(^|_)(r2|mse|rmse|mae|mape)(_|$)", re.IGNORECASE)

check_results = []


def check(label, passed):
    """Record one pass/fail result and print it."""
    passed = bool(passed)
    check_results.append(passed)
    print("  [{}] {}".format("PASS" if passed else "FAIL", label))


def header(title):
    print("\n{}\n{}".format(title, "-" * len(title)))


def quietly(function, *args):
    """Run project code without its progress output, showing it only on failure."""
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            return function(*args)
    except BaseException:
        print(captured.getvalue())
        raise


def redirect_project_outputs(root):
    """Send results and the feature cache to `root`, for this process only."""
    dataset, _, tee, _ = paths.paths().dataset_files()
    cache_dir, results_dir = str(root / "cache"), str(root / "results")
    paths.paths.dataset_files = lambda self: (dataset, cache_dir, tee, results_dir)


# ------------------------------------------------------------ 1. source code
def scan_source(text):
    """List the estimators a script builds and what it imports from scikit-learn."""
    built, imported = Counter(), set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sklearn"):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name.endswith(("Classifier", "Regressor")):
                built[name] += 1
    return built, imported


def source_verdict(built, imported):
    regression_imports = {
        name for name in imported
        if name.endswith("Regressor") or name in REGRESSION_METRICS
    }
    if regression_imports or any(name.endswith("Regressor") for name in built):
        return "REGRESSION"
    if built and all(name.endswith("Classifier") for name in built):
        return "CLASSIFICATION"
    return "UNCLEAR"


def check_source():
    header("1. SOURCE CODE")
    for filename in ("decision_tree.py", "random_forest_baseline.py"):
        text = (HERE / filename).read_text(encoding="utf-8")
        built, imported = scan_source(text)
        print("  {}: builds {}".format(filename, dict(built)))
        print("    scikit-learn imports: {}".format(sorted(imported)))
        check("{} uses only classifiers, with no regressor or regression metric".format(
            filename), source_verdict(built, imported) == "CLASSIFICATION")
        # Prove this scan can fail: an in-memory copy using regressors must not pass.
        changed = scan_source(text.replace("Classifier", "Regressor"))
        check("the same scan says REGRESSION for an in-memory copy using regressors",
              source_verdict(*changed) == "REGRESSION")


# --------------------------------------------------------- 2. the fitted models
class RunLog:
    """What one run of a project script passed to fit() and got from predict()."""

    def __init__(self):
        self.fit_count = 0
        self.fit_target_types = Counter()
        self.fit_labels = set()
        self.predicted_rows = 0
        self.predict_types = Counter()
        self.predicted_values = set()
        self.final_model = None
        self.last_inputs = None


def spy_on(base, log):
    """Return a subclass of `base` that behaves the same but records its fit/predict."""

    class Spy(base):
        def fit(self, X, y, *args, **kwargs):
            fitted = super().fit(X, y, *args, **kwargs)
            log.fit_count += 1
            log.fit_target_types[type_of_target(y)] += 1
            log.fit_labels.update(np.unique(y).tolist())
            log.final_model = self   # the scripts fit their final model last
            return fitted

        def predict(self, X):
            output = super().predict(X)
            log.predicted_rows += len(output)
            log.predict_types[type_of_target(output)] += 1
            log.predicted_values.update(np.unique(output).tolist())
            log.last_inputs = np.array(X[:5000])
            return output

    Spy.__name__ = Spy.__qualname__ = base.__name__
    return Spy


def run_tree(task_key, folder, engines, out_root):
    """Run decision_tree.run_task with the arguments its main() uses for this task."""
    net_engine, cell_module = engines
    if task_key == "net":
        task, engine, feature_set = net_engine.NetDelayTask(), net_engine, "full"
    else:
        task, engine, feature_set = cell_module.CellDelayTask(), cell_module.km, "lut_sta_ctx"
    quietly(
        dt.run_task, folder, task, engine, feature_set,
        list(net_engine.QUICK_TRAIN), list(net_engine.QUICK_TEST), out_root,
    )


def run_forest(task_key, out_root):
    """Run the forest script's own command line, using its documented --quick mode."""
    arguments = sys.argv
    sys.argv = [
        "random_forest_baseline.py", "--quick", "--task", task_key, "--seeds", str(SEED),
        "--jobs", FOREST_JOBS, "--out", str(out_root),
    ]
    try:
        quietly(rf.main)
    finally:
        sys.argv = arguments


def describe(model, X):
    """Inspect any fitted scikit-learn tree or forest and decide what kind it is.

    The same function is used for the project's models and for the controls.
    """
    output = np.asarray(model.predict(X))
    classes = getattr(model, "classes_", None)
    can_give_probabilities = hasattr(model, "predict_proba")
    says_classifier, says_regressor = is_classifier(model), is_regressor(model)

    if (says_classifier and not says_regressor and classes is not None
            and can_give_probabilities and np.isin(output, classes).all()):
        verdict = "CLASSIFICATION"
    elif (says_regressor and not says_classifier and classes is None
            and not can_give_probabilities):
        verdict = "REGRESSION"
    else:
        verdict = "UNCLEAR"

    forest_trees = getattr(model, "estimators_", None)
    tree = forest_trees[0] if forest_trees is not None else model
    first_probabilities = "-"
    if can_give_probabilities:
        first_probabilities = np.array2string(
            model.predict_proba(X[:1])[0], precision=2, suppress_small=True
        )
    return {
        "scikit-learn model": type(model).__name__,
        "trees inside": (
            "{} x {}".format(len(forest_trees), type(tree).__name__)
            if forest_trees is not None else "-"
        ),
        "is_classifier()": str(says_classifier),
        "is_regressor()": str(says_regressor),
        "classes_": str(classes.tolist()) if classes is not None else "-",
        "has predict_proba": str(can_give_probabilities),
        "split criterion": str(model.criterion),
        "values stored per tree node": str(tree.tree_.value.shape[-1]),
        "predict() dtype": str(output.dtype),
        "predict() target type": type_of_target(output),
        "distinct predict() values": str(len(np.unique(output))),
        "first predict() values": np.array2string(output[:4], precision=4),
        "first predict_proba row": first_probabilities,
        "VERDICT": verdict,
    }


def print_table(described):
    """Print one column per model and one row per inspected property."""
    names = list(described)
    fields = list(described[names[0]])
    label_width = max(len(field) for field in fields)
    widths = [
        max(len(name), *(len(described[name][field]) for field in fields))
        for name in names
    ]
    print("  " + " " * label_width + "  " + "  ".join(
        name.ljust(width) for name, width in zip(names, widths)))
    for field in fields:
        print("  " + field.ljust(label_width) + "  " + "  ".join(
            described[name][field].ljust(width) for name, width in zip(names, widths)))


def check_run_log(name, log):
    """Check everything one run sent to fit() and received from predict()."""
    class_ids = set(range(len(CLASS_NAMES)))
    print("  {}: {} fits, targets {}; {:,} rows predicted, values {}".format(
        name, log.fit_count, dict(log.fit_target_types), log.predicted_rows,
        sorted(log.predicted_values)))
    check("{}: every fit was on class IDs 0-4, never on raw delay".format(name),
          log.fit_count > 0 and log.fit_labels == class_ids
          and set(log.fit_target_types) <= {"binary", "multiclass"})
    check("{}: every predict() returned only class IDs 0-4".format(name),
          log.predicted_rows > 0 and log.predicted_values <= class_ids
          and set(log.predict_types) <= {"binary", "multiclass"})


# --------------------------------------------------------------- 3. output files
def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def column(rows, name):
    return np.array([float(row[name]) for row in rows])


def check_metric_columns(name, metrics):
    columns = set(metrics[0])
    check("{}: metrics.csv has accuracy and F1, and no R2/MSE/MAE columns".format(name),
          {"accuracy", "macro_f1", "weighted_f1"} <= columns
          and not any(REGRESSION_COLUMN.search(item) for item in columns))


def check_tree_files(name, task_dir):
    """decision_tree.py saves one row per sample, so check the predictions themselves."""
    rows = read_csv(task_dir / "pooled_predictions.csv")
    metrics = read_csv(task_dir / "metrics.csv")
    pooled = next(row for row in metrics if row["design"] == "POOLED")
    edges = [float(pooled["edge_{}_upper_ns".format(label)]) for label in CLASS_NAMES[:-1]]
    true_class, predicted = column(rows, "true_class"), column(rows, "predicted_class")
    delay = column(rows, "delay_ns")
    probabilities = np.column_stack(
        [column(rows, "prob_{}".format(label)) for label in CLASS_NAMES]
    )

    print("  {}: pooled_predictions.csv has {:,} rows and these columns:".format(name, len(rows)))
    print("    {}".format(", ".join(rows[0])))
    example = rows[0]
    print("    example: delay_ns={:.6f} true={} predicted={} prob_{}={:.3f}".format(
        float(example["delay_ns"]), example["true_class_name"],
        example["predicted_class_name"], example["predicted_class_name"],
        float(example["prob_{}".format(example["predicted_class_name"])])))

    whole_numbers = all(re.fullmatch("[0-4]", row["predicted_class"]) for row in rows)
    check("{}: every predicted_class is a whole number from 0 to 4".format(name), whole_numbers)
    check("{}: every predicted_class_name is the matching C1-C5 label".format(name),
          whole_numbers and all(
              row["predicted_class_name"] == CLASS_NAMES[int(row["predicted_class"])]
              for row in rows))
    check("{}: the 5 class probabilities add up to 1 on every row".format(name),
          np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5))
    check("{}: predicted_class is the most probable class on every row".format(name),
          (probabilities.argmax(axis=1) == predicted).all())
    check("{}: delay_ns is the true delay, not a prediction (it bins to true_class)".format(name),
          (np.digitize(delay, edges) == true_class).all())
    check_metric_columns(name, metrics)
    macro_f1 = f1_score(
        true_class, predicted, labels=range(len(CLASS_NAMES)),
        average="macro", zero_division=0,
    )
    check("{}: POOLED accuracy and macro F1 recompute from the predicted classes".format(name),
          np.isclose((true_class == predicted).mean(), float(pooled["accuracy"]))
          and np.isclose(macro_f1, float(pooled["macro_f1"])))


def check_forest_files(name, task_dir):
    """random_forest_baseline.py saves no per-sample rows, so check its summaries."""
    metrics = read_csv(task_dir / "metrics.csv")
    confusion_rows = read_csv(task_dir / "confusion_matrix.csv")
    metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    pooled = next(row for row in metrics if row["design"] == "POOLED")
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for row in confusion_rows:
        matrix[CLASS_NAMES.index(row["true_class"]),
               CLASS_NAMES.index(row["predicted_class"])] = int(row["count"])

    print("  {}: files written: {}".format(
        name, ", ".join(sorted(item.name for item in task_dir.iterdir()))))
    check_metric_columns(name, metrics)
    check("{}: confusion_matrix.csv is a full C1-C5 by C1-C5 table".format(name),
          len(confusion_rows) == len(CLASS_NAMES) ** 2)
    check("{}: POOLED sample count and accuracy recompute from the confusion matrix".format(name),
          matrix.sum() == int(pooled["n_samples"])
          and np.isclose(np.trace(matrix) / matrix.sum(), float(pooled["accuracy"])))
    check("{}: metadata.json lists 4 class boundaries, which make 5 classes".format(name),
          len(metadata["class_edges_ns"]) == len(CLASS_NAMES) - 1)


# ------------------------------------------------------------------ 4. controls
def run_controls(net_engine):
    """Train real regressors on real data; the inspection must call them REGRESSION.

    The class-ID controls catch the sneakier mistake of a regressor trained on
    the labels 0-4, which could look like classification from its target alone.
    """
    task = net_engine.NetDelayTask()
    X, delay = dt._load_design(task, net_engine, "wbqspiflash", list(task.featsets["full"]))
    class_ids = np.digitize(delay, np.quantile(delay, [0.2, 0.4, 0.6, 0.8]))
    tree_settings = {"max_depth": 6, "random_state": SEED}
    forest_settings = dict(tree_settings, n_estimators=10, n_jobs=int(FOREST_JOBS))
    models = {
        "tree/delay": DecisionTreeRegressor(**tree_settings).fit(X, delay),
        "forest/delay": RandomForestRegressor(**forest_settings).fit(X, delay),
        "tree/class-IDs": DecisionTreeRegressor(**tree_settings).fit(X, class_ids),
        "forest/class-IDs": RandomForestRegressor(**forest_settings).fit(X, class_ids),
    }
    return {name: describe(model, X[:5000]) for name, model in models.items()}


# ------------------------------------------------------------------------- main
def main():
    started = time.time()
    check_source()

    with tempfile.TemporaryDirectory(
        prefix="tree_forest_check_", ignore_cleanup_errors=True
    ) as temporary:
        root = Path(temporary)
        redirect_project_outputs(root)
        net_file, cell_file = paths.paths().dependency_paths()
        net_engine = quietly(dt._load_module, "check_net_kmeans", net_file)
        cell_module = quietly(dt._load_module, "check_cell_kmeans", cell_file)
        tree_root, forest_root = root / "decision_tree", root / "random_forest"
        tree_root.mkdir()
        forest_root.mkdir()

        header("Running both scripts on the 7 --quick designs")
        tree_class, forest_class = dt.DecisionTreeClassifier, rf.RandomForestClassifier
        logs = {}
        for task_key, folder in TASKS:
            name = "tree/{}".format(task_key)
            print("  {} ...".format(name), flush=True)
            logs[name] = RunLog()
            dt.DecisionTreeClassifier = spy_on(tree_class, logs[name])
            run_tree(task_key, folder, (net_engine, cell_module), tree_root)
        for task_key, folder in TASKS:
            name = "forest/{}".format(task_key)
            print("  {} ...".format(name), flush=True)
            logs[name] = RunLog()
            rf.RandomForestClassifier = spy_on(forest_class, logs[name])
            run_forest(task_key, forest_root)

        header("2. FITTED MODELS FROM THOSE RUNS")
        for name, log in logs.items():
            check_run_log(name, log)
        print()
        described = {
            name: describe(log.final_model, log.last_inputs) for name, log in logs.items()
        }
        print_table(described)
        print()
        for name, facts in described.items():
            check("{}: final model is CLASSIFICATION".format(name),
                  facts["VERDICT"] == "CLASSIFICATION")

        header("3. FILES THE SCRIPTS WROTE")
        for task_key, folder in TASKS:
            check_tree_files("tree/{}".format(task_key), tree_root / folder)
        for task_key, folder in TASKS:
            check_forest_files("forest/{}".format(task_key), forest_root / folder)

        header("4. CONTROLS: REAL REGRESSORS THROUGH THE SAME INSPECTION")
        controls = run_controls(net_engine)
        print_table(controls)
        print()
        for name, facts in controls.items():
            check("{}: control is reported as REGRESSION".format(name),
                  facts["VERDICT"] == "REGRESSION")

    header("RESULT")
    for label, prefix in (("Decision tree", "tree/"), ("Random forest", "forest/")):
        verdicts = sorted({
            facts["VERDICT"] for name, facts in described.items() if name.startswith(prefix)
        })
        print("  {}: {}".format(label, " / ".join(verdicts)))
    print("  Checks passed: {} of {} ({:.0f} s)".format(
        sum(check_results), len(check_results), time.time() - started))
    return 0 if all(check_results) else 1


if __name__ == "__main__":
    sys.exit(main())
