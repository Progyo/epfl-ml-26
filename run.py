import numpy as np

from cleaner import add_interactions, clean_data, interaction_pairs
from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression_newton, compute_logistic_f1s, build_k_indices,f1_score, compute_logistic_loss

from time import time

DATA_DIR = "dataset"

# These parameters were determined by exeperiments run in main.py
LAMBDA_ = 0.01
THRESHOLD = 0.250
NEWTON_ITERS = 10

K_FOLD = 5
SEED = 42

def cross_validate(y, x, k_fold=K_FOLD, seed=SEED):
    # Very similar to the exercise code
    # Just build K folds and calculate the mean f1 and loss
    k_indices = build_k_indices(y, k_fold, seed)
    f1s, losses = [], []
    for k in range(k_fold):
        te, tr = k_indices[k], np.delete(k_indices, k, axis=0).ravel()
        x_tr, x_te = x[tr], x[te]
        x_tr, x_te = add_interactions(x_tr, x_te, interaction_pairs(y[tr], x_tr))
        w, _ = reg_logistic_regression_newton(y[tr], x_tr, LAMBDA_, np.zeros(x_tr.shape[1]), NEWTON_ITERS)
        p = 1 / (1 + np.exp(-x_te @ w))
        f1s.append(f1_score(y[te], (p >= THRESHOLD).astype(int)))
        losses.append(compute_logistic_loss(y[te], x_te, w))
        print(f"fold {k + 1}/{k_fold}: F1={f1s[-1]:.4f}, loss={losses[-1]:.4f}")
    return np.mean(f1s), np.mean(losses)

def main():


    
    now = time()

    # Load CSV using helper ffunctions supplied
    print('Loading CSV')
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)

    print(f'Took {(time() - now):.2f} seconds')
    now = time()

    # Clean our data using the methods developed in cleaner.py and main.py
    print('Cleaning Data')
    x_train, x_test, y_train = clean_data(x_train, x_test, y_train)
    
    print(f'Took {(time() - now):.2f} seconds')
    now = time()

    # CV runs before the interactions are added, since each fold picks its own interaction columns
    # cv_f1, cv_loss = cross_validate(y_train, x_train)
    # print(f"{K_FOLD}-fold CV: F1={cv_f1:.4f}, loss={cv_loss:.4f}")

    # During tests we discovered that adding additional cross interactions between inputs improved performance
    print('Adding interactions')
    x_train, x_test = add_interactions(x_train, x_test, interaction_pairs(y_train, x_train))

    print(f'Took {(time() - now):.2f} seconds')
    now = time()

    #### UNCOMMENT THIS SECTION TO TRAIN MODEL
    
    # print('Training model weights')
    # w, _ = reg_logistic_regression_newton(y_train, x_train, LAMBDA_, np.zeros(x_train.shape[1]), NEWTON_ITERS)

    # print(f'Took {(time() - now):.2f} seconds')
    # now = time()

    # print('Saving model weights')
    # np.save('model_weights.npy', w)

    #### END OF TRAINING CODE

    #### Load model from file
    print('Loading model weights')
    w = np.load('model_weights.npy')

    print(f'Took {(time() - now):.2f} seconds')
    now = time()
    ####

    print('Creating submission.csv')
    y_pred = 1 / (1 + np.exp(-x_test @ w)) >= THRESHOLD    
    create_csv_submission(test_ids, np.where(y_pred, 1, -1), "submission.csv")
    print(f'Took {(time() - now):.2f} seconds')
    now = time()

if __name__ == "__main__":
    main()
