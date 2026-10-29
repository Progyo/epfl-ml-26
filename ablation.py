"""Ablation study: run.py's final model and hyperparameters, trained on main.py's 80/20 split, with steps of the
pipeline switched off.

    python ablation.py                       # the final model
    python ablation.py --no-interactions     # one step off; the flags can be combined
    python ablation.py --all                 # the final model, then each step off on its own, then the old cleaning

Each setting is scored on the validation rows: F1 and accuracy at the cutoff (run.py's THRESHOLD unless --cutoff is
given), and F1 at the cutoff that maximizes F1 on the training rows, since some steps move the best cutoff. Every
setting keeps run.py's lambda and number of Newton steps.
"""

import argparse
from time import time

import numpy as np

import old_cleaner
from cleaner import add_interactions, clean_data, interaction_pairs
from helpers import load_csv_data
from implementations import compute_logistic_f1s, f1_score, reg_logistic_regression_newton
from main import DATA_DIR, accuracy, split_data
from run import LAMBDA_, NEWTON_ITERS, THRESHOLD

CUTOFFS = np.linspace(0.05, 0.95, 91)  # searched on the training rows

# The final model's pipeline; each flag below changes one setting
FINAL = {
    "interactions": True,
    "codebook": True,
    "max_nan": 1,
    "one_hot_encoding": True,
    "missing_indicators": True,
    "old_cleaner": False,
}
# (command-line flag, setting, value with the flag)
FLAGS = [
    ("no_interactions", "interactions", False),
    ("no_codebook", "codebook", False),
    ("drop_half_missing", "max_nan", 0.5),
    ("no_one_hot", "one_hot_encoding", False),
    ("no_missing_indicators", "missing_indicators", False),
    ("old_cleaner", "old_cleaner", True),
]
# --all: a label and the settings that differ from FINAL, for each row
ROWS = [
    ("final model", {}),
    ("- interactions", {"interactions": False}),
    ("- codebook recoding", {"codebook": False}),
    ("+ drop columns >= 50% missing", {"max_nan": 0.5}),
    ("- one-hot encoding", {"one_hot_encoding": False}),
    ("- missing indicators", {"missing_indicators": False}),
    ("old cleaning baseline", {"old_cleaner": True, "interactions": False}),
]


def sigmoid(z):
    # exp overflows to inf for very negative scores, giving 0
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-z))


def evaluate(x_train, x_test, y_train, settings):
    """Train run.py's model on main.py's training split, with the pipeline given by settings.

    Returns the number of columns, the validation labels and probabilities, and the cutoff that maximizes F1 on the
    training rows.
    """
    if settings["old_cleaner"]:
        x, x_te, y = old_cleaner.clean_data(x_train, x_test, y_train)
    else:
        x, x_te, y = clean_data(
            x_train,
            x_test,
            y_train,
            codebook=settings["codebook"],
            max_nan=settings["max_nan"],
            one_hot_encoding=settings["one_hot_encoding"],
            missing_indicators=settings["missing_indicators"],
        )
    # The test rows play no part here
    del x_te
    x_tr, y_tr, x_val, y_val = split_data(x, y)
    del x
    if settings["interactions"]:
        # Picked on the training split only, as in main.py
        x_tr, x_val = add_interactions(x_tr, x_val, interaction_pairs(y_tr, x_tr))
    w, _ = reg_logistic_regression_newton(y_tr, x_tr, LAMBDA_, np.zeros(x_tr.shape[1]), NEWTON_ITERS)
    with np.errstate(over="ignore"):
        best_cutoff = CUTOFFS[np.argmax(compute_logistic_f1s(y_tr, x_tr, w, CUTOFFS))]
    return x_tr.shape[1], y_val, sigmoid(x_val @ w), best_cutoff


def report(label, n_columns, y_val, p_val, cutoff, best_cutoff, seconds, reference_f1=None):
    """Print one row of the ablation, with the change in F1 from reference_f1 if given, and return its F1."""
    y_pred = (p_val >= cutoff).astype(int)
    f1 = f1_score(y_val, y_pred)
    change = "" if reference_f1 is None else f" ({f1 - reference_f1:+.4f})"
    best_f1 = f1_score(y_val, (p_val >= best_cutoff).astype(int))
    print(
        f"{label}: {n_columns} columns, val F1={f1:.4f}{change}, val accuracy={accuracy(y_val, y_pred):.4f} at "
        f"cutoff {cutoff:g}; val F1={best_f1:.4f} at the best training cutoff {best_cutoff:.2f} ({seconds:.0f}s)",
        flush=True,
    )
    return f1


def main(rows, cutoff):
    """Score each (label, settings changed from FINAL) of rows. With several rows, the changes in F1 are relative
    to the first, and the first is also scored at cutoff 0.5 to show what tuning the cutoff adds."""
    x_train, x_test, y_train, _, _ = load_csv_data(DATA_DIR)
    reference_f1 = None
    for label, changes in rows:
        start = time()
        n_columns, y_val, p_val, best_cutoff = evaluate(x_train, x_test, y_train, {**FINAL, **changes})
        f1 = report(label, n_columns, y_val, p_val, cutoff, best_cutoff, time() - start, reference_f1)
        if len(rows) > 1 and reference_f1 is None:
            reference_f1 = f1
            report(f"{label}, cutoff 0.5", n_columns, y_val, p_val, 0.5, best_cutoff, 0, reference_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-interactions", action="store_true", help="no pairwise interaction columns")
    parser.add_argument("--no-codebook", action="store_true", help="recode answer codes by the width rule alone")
    parser.add_argument(
        "--drop-half-missing",
        action="store_true",
        help="drop the columns with half or more of their training values missing",
    )
    parser.add_argument(
        "--no-one-hot", action="store_true", help="keep categorical columns as their rank instead of one-hot encoding"
    )
    parser.add_argument(
        "--no-missing-indicators", action="store_true", help="only impute missing values, without 0/1 missing columns"
    )
    parser.add_argument(
        "--old-cleaner",
        action="store_true",
        help="clean with old_cleaner.py (the 2026-10-20 pipeline); combine with --no-interactions for it as it was",
    )
    parser.add_argument(
        "--cutoff", type=float, default=THRESHOLD, help=f"cutoff on the predicted probability (default {THRESHOLD})"
    )
    parser.add_argument(
        "--all", action="store_true", help="the final model, each step off on its own, and the old cleaning, in one go"
    )
    args = parser.parse_args()

    changes = {setting: value for flag, setting, value in FLAGS if getattr(args, flag)}
    if args.all and changes:
        parser.error("--all runs every setting itself; give it no other flags")
    if args.old_cleaner and set(changes) - {"old_cleaner", "interactions"}:
        parser.error("the cleaning flags only apply to cleaner.py, not to --old-cleaner")
    label = ", ".join("--" + flag.replace("_", "-") for flag, _, _ in FLAGS if getattr(args, flag)) or "final model"
    main(ROWS if args.all else [(label, changes)], args.cutoff)
