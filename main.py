from functools import partial

import numpy as np

from cleaner import clean_data
from helpers import load_csv_data, create_csv_submission
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)

DATA_DIR = "dataset"
SEED = 42
VAL_RATIO = 0.2
MAX_ITERS = 200
GAMMA = 0.01
LAMBDA_ = 0.2

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
]


def split_data(x, y, val_ratio=VAL_RATIO, seed=SEED):
    """Randomly split (x, y) into a training and a validation set."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def predict(x, w, is_logistic):
    """Predict labels in {0, 1} from features x and weights w."""
    if is_logistic:
        return (1 / (1 + np.exp(-x @ w)) >= 0.5).astype(int)
    return (x @ w >= 0.5).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred)


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)

    x_train, x_test, y_train = clean_data(x_train, x_test, y_train)

    x_tr, y_tr, x_val, y_val = split_data(x_train, y_train)

    best_name, best_fn, best_is_logistic, best_acc = None, None, None, -np.inf
    for name, fn, is_logistic in MODELS:
        initial_w = np.zeros(x_tr.shape[1])
        w, loss = fn(y_tr, x_tr, initial_w)
        val_acc = accuracy(y_val, predict(x_val, w, is_logistic))
        print(f"{name}: train loss={loss:.4f}, val accuracy={val_acc:.4f}")

        if val_acc > best_acc:
            best_name, best_fn, best_is_logistic, best_acc = name, fn, is_logistic, val_acc

    print(f"best model: {best_name} (val accuracy={best_acc:.4f})")

    initial_w = np.zeros(x_train.shape[1])
    w, _ = best_fn(y_train, x_train, initial_w)
    y_pred = predict(x_test, w, best_is_logistic)
    create_csv_submission(test_ids, np.where(y_pred == 1, 1, -1), "submission.csv")


if __name__ == "__main__":
    main()
