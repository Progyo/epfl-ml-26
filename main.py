import numpy as np

from cleaner import clean_data
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression

DATA_DIR = "dataset"
SEED = 42
VAL_RATIO = 0.2
MAX_ITERS = 100
GAMMA = 0.1


def split_data(x, y, val_ratio=VAL_RATIO, seed=SEED):
    """Randomly split (x, y) into a training and a validation set."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def predict(x, w):
    """Predict labels in {0, 1} from features x and weights w."""
    return (1 / (1 + np.exp(-x @ w)) >= 0.5).astype(int)


def accuracy(y, y_pred):
    return np.mean(y == y_pred)


def main():
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_DIR)

    x_train, x_test, y_train = clean_data(x_train, x_test, y_train)

    x_tr, y_tr, x_val, y_val = split_data(x_train, y_train)

    initial_w = np.zeros(x_tr.shape[1])
    w, loss = logistic_regression(y_tr, x_tr, initial_w, MAX_ITERS, GAMMA)
    print(f"train loss: {loss:.4f}, val accuracy: {accuracy(y_val, predict(x_val, w)):.4f}")

    initial_w = np.zeros(x_train.shape[1])
    w, _ = logistic_regression(y_train, x_train, initial_w, MAX_ITERS, GAMMA)
    y_pred = predict(x_test, w)
    create_csv_submission(test_ids, y_pred, "submission.csv")


if __name__ == "__main__":
    main()
