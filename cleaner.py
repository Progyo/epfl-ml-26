import numpy as np


def clean_data(x_train, x_test, y_train):
    """Clean and preprocess the raw dataset.

    Args:
        x_train: shape=(N, D). Raw training features, as returned by load_csv_data.
        x_test: shape=(M, D). Raw test features, as returned by load_csv_data.
        y_train: shape=(N,). Raw training labels, as returned by load_csv_data.

    Returns:
        x_train: shape=(N, D'). Cleaned/processed training features.
        x_test: shape=(M, D'). Cleaned/processed test features, using the
            same transformation (e.g. same normalization stats) as x_train.
        y_train: shape=(N,). Labels, recoded to {0, 1} if needed.
    """
    raise NotImplementedError
