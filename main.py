import argparse
from functools import partial

import numpy as np

import old_cleaner
from cleaner import add_interactions, clean_data, interaction_pairs
from helpers import load_csv_data, create_csv_submission
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
    reg_logistic_regression_newton,
    cross_fold_reg_logistic_regression_newton,
    f1_score,
)

DATA_DIR = "dataset"
SEED = 42
VAL_RATIO = 0.2
MAX_ITERS = 200
NEWTON_ITERS = 10
GAMMA = 0.01
LAMBDA_ = 0.2
K_FOLD = 5
LAMBDAS = np.logspace(-6, -1, 6)
THRESHOLDS = np.linspace(0.05, 0.95, 19)
DEFAULT_THRESHOLD = 0.5

MODELS = [
    ("mean_squared_error_gd", partial(mean_squared_error_gd, max_iters=MAX_ITERS, gamma=GAMMA), False),
    ("mean_squared_error_sgd", partial(mean_squared_error_sgd, max_iters=MAX_ITERS, gamma=GAMMA), False),
    ("least_squares", lambda y, tx, initial_w: least_squares(y, tx), False),
    ("ridge_regression", lambda y, tx, initial_w: ridge_regression(y, tx, LAMBDA_), False),
    ("logistic_regression", partial(logistic_regression, max_iters=MAX_ITERS, gamma=GAMMA), True),
    (
        "reg_logistic_regression",
        lambda y, tx, initial_w: reg_logistic_regression(y, tx, LAMBDA_, initial_w, MAX_ITERS, GAMMA),
        True,
    ),
    (
        "reg_logistic_regression_newton",
        lambda y, tx, initial_w: reg_logistic_regression_newton(y, tx, LAMBDA_, initial_w, NEWTON_ITERS),
        True,
    ),
    (
        "cross_fold_reg_logistic_regression_newton",
        lambda y, tx, initial_w: cross_fold_reg_logistic_regression_newton(
            y, tx, initial_w, K_FOLD, LAMBDAS, THRESHOLDS, NEWTON_ITERS, SEED
        ),
        True,
    ),
]


def split_data(x, y, val_ratio=VAL_RATIO, seed=SEED):
    """Randomly split (x, y) into a training and a validation set."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def fit(fn, y, tx):
    """Train fn from zero weights and return (w, loss, threshold).

    Models that tune their own cutoff return it as a third value; the others get DEFAULT_THRESHOLD.
    """
    result = fn(y, tx, np.zeros(tx.shape[1]))
    threshold = result[2] if len(result) == 3 else DEFAULT_THRESHOLD
    return result[0], result[1], threshold


def predict(x, w, is_logistic, threshold=DEFAULT_THRESHOLD):
    """Predict labels in {0, 1} from features x and weights w."""
    if is_logistic:
        return (1 / (1 + np.exp(-x @ w)) >= threshold).astype(int)
    return (x @ w >= threshold).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred)


def main(use_old_cleaner=False):
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)

    clean = old_cleaner.clean_data if use_old_cleaner else clean_data
    x_train, x_test, y_train = clean(x_train, x_test, y_train)

    x_tr, y_tr, x_val, y_val = split_data(x_train, y_train)
    if not use_old_cleaner:
        # Interaction columns picked on the training split only, so the validation labels play no part in them.
        x_tr, x_val = add_interactions(x_tr, x_val, interaction_pairs(y_tr, x_tr))

    best_name, best_fn, best_is_logistic, best = None, None, None, -np.inf
    for name, fn, is_logistic in MODELS:
        w, loss, threshold = fit(fn, y_tr, x_tr)
        y_val_pred = predict(x_val, w, is_logistic, threshold)
        val_acc = accuracy(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        print(
            f"{name}: train loss={loss:.4f}, cutoff={threshold:.2f}, "
            f"val accuracy={val_acc:.4f}, val F1={val_f1:.4f}"
        )

        if val_f1 > best:
            best_name, best_fn, best_is_logistic, best = name, fn, is_logistic, val_f1

    print(f"best model: {best_name} (val f1={best:.4f})")
    del x_tr, x_val

    # Interaction columns picked again on all training rows. The cross-fold model re-tunes lambda and its cutoff
    # on the full training set here.
    if not use_old_cleaner:
        x_train, x_test = add_interactions(x_train, x_test, interaction_pairs(y_train, x_train))
    w, _, threshold = fit(best_fn, y_train, x_train)
    y_pred = predict(x_test, w, best_is_logistic, threshold)
    submission = "submission_old_cleaner.csv" if use_old_cleaner else "submission.csv"
    create_csv_submission(test_ids, np.where(y_pred == 1, 1, -1), submission)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all models on a validation split and write a submission.")
    parser.add_argument(
        "--old-cleaner",
        action="store_true",
        help="clean with old_cleaner.py (the 2026-10-20 pipeline, no interaction columns) as a baseline",
    )
    main(parser.parse_args().old_cleaner)
