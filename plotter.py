r"""Figures for the report, written to figures/ as PDFs.

    python plotter.py

Each experiment's results are cached in figures/cache/, so running the script again only redraws the figures; delete
a cache file to rerun its experiment. Without a cache the run takes about an hour (the 3000 GD steps of the training
experiment, then the lambda sweep) and peaks at about 9 GB of RAM, as ridge_regression makes a float64 copy of the
training matrix to compute its loss.
"""

import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cleaner import add_interactions, clean_data, interaction_pairs
from helpers import load_csv_data
from implementations import (
    build_k_indices,
    compute_logistic_loss,
    compute_loss,
    f1_score,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    reg_logistic_regression_newton,
    ridge_regression,
)
from main import split_data
from run import K_FOLD, LAMBDA_, NEWTON_ITERS, SEED, THRESHOLD

DATA_DIR = "dataset"
FIG_DIR = "figures"
CACHE_DIR = os.path.join(FIG_DIR, "cache")

GD_ITERS = 1000
GD_CHECKPOINTS = np.unique(np.round(np.geomspace(1, GD_ITERS, 30)).astype(int))  # log-spaced, like the x-axis
LAMBDAS = np.union1d(np.logspace(-6, 1, 15), [LAMBDA_])  # half decades, plus run.py's lambda
NEWTON_TOL = 1e-4  # lambda sweep: stop once a Newton step moves w by less than this, relative to |w|
CUTOFFS = np.linspace(0.01, 0.8, 80)  # cutoff plot; above 0.8 too few rows are predicted 1 for a stable precision

# The course's LaTeX template is IEEEtran: a column is 252pt wide, the page 516pt, and captions are 8pt Times
COLUMN_WIDTH = 252 / 72.27  # inches
TEXT_WIDTH = 516 / 72.27
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, REFERENCE, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],  # ships with matplotlib, close to Times
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.3,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
    "pdf.fonttype": 42,  # TrueType: conference PDF checkers reject matplotlib's default Type 3 fonts
}
# Training curves: legend label and color of each model; convergence.pdf's two curves keep their colors
CURVES = {
    "mse_gd": ("MSE, GD", YELLOW),
    "mse_sgd": ("MSE, SGD", MAGENTA),
    "logistic_gd": ("Logistic, GD", AQUA),
    "reg_logistic_gd": ("Reg. logistic, GD", BLUE),
    "newton": ("Reg. logistic, Newton (final)", ORANGE),
}
NEWTON_MARKERS = {"marker": "o", "markersize": 3.5, "markeredgecolor": "white", "markeredgewidth": 0.5}


def sigmoid(z):
    with np.errstate(over="ignore"):  # exp overflows to inf for very negative scores, giving 0
        return 1 / (1 + np.exp(-z))


def best_f1(y, score):
    """Highest F1 over all cutoffs on score. Predicting 1 for the k highest scores gives F1 = 2 TP_k / (k + P)."""
    tp = np.cumsum(y[np.argsort(-score)])
    return np.max(2 * tp / (np.arange(1, len(y) + 1) + tp[-1]))


def precision_recall_f1(y, p, cutoffs):
    """Precision, recall and F1 of predicting 1 where p >= cutoff, for each cutoff (precision is NaN when nothing is
    predicted 1)."""
    pred = p[:, None] >= cutoffs
    tp = (pred & (y[:, None] == 1)).sum(axis=0)
    n_pred, n_pos = pred.sum(axis=0), y.sum()
    with np.errstate(invalid="ignore"):
        return tp / n_pred, tp / n_pos, 2 * tp / (n_pred + n_pos)


def largest_eigenvalue(tx, iters=50):
    """Largest eigenvalue of tx^T tx / N, by power iteration. It is the Lipschitz constant L of the MSE's gradient;
    the mean NLL's is a quarter of it (sigma' <= 1/4), plus 2 lambda_ with a penalty lambda_ ||w||^2."""
    v = np.random.default_rng(SEED).standard_normal(tx.shape[1]).astype(tx.dtype)
    for _ in range(iters):
        v = tx.T @ (tx @ v) / len(tx)
        top = np.linalg.norm(v)
        v /= top
    return top


def newton_fit(y, tx, lambda_, w):
    """reg_logistic_regression_newton from w, one step at a time, until a step moves w by less than NEWTON_TOL
    relative to |w| (at most NEWTON_ITERS steps). Returns w and the number of steps taken."""
    for step in range(1, NEWTON_ITERS + 1):
        w_next, _ = reg_logistic_regression_newton(y, tx, lambda_, w, 1)
        moved = np.linalg.norm(w_next - w) / np.linalg.norm(w_next)
        w = w_next
        if moved < NEWTON_TOL:
            break
    return w, step


def load_data():
    """The cleaned training rows (float32 features, 0/1 labels), as in run.py."""
    x_train, x_test, y_train, _, _ = load_csv_data(DATA_DIR)
    x_train, _, y_train = clean_data(x_train, x_test, y_train)
    return x_train, y_train


def training_experiment(x, y):
    """Held-out loss and best-cutoff F1 along the training of each iterative model, from w = 0 on main.py's split
    (the features plus interaction columns picked on the training split).

    GD runs GD_ITERS steps of size 1/L, SGD one pass over the rows with step 1 / max_n |x_n|^2 (the largest step that
    never overshoots a row), and Newton NEWTON_ITERS steps. The regularized models use run.py's LAMBDA_, so
    reg_logistic_regression and Newton minimize the final model's objective.
    """
    x_tr, y_tr, x_val, y_val = split_data(x, y)
    x_tr, x_val = add_interactions(x_tr, x_val, interaction_pairs(y_tr, x_tr))
    # float32 labels and weights: with int64 or float64 ones, numpy would copy x_tr to float64 at every product
    y_tr = y_tr.astype(x_tr.dtype)
    w0 = np.zeros(x_tr.shape[1], dtype=x_tr.dtype)
    top = largest_eigenvalue(x_tr)
    sgd_gamma = 1 / np.einsum("ij,ij->i", x_tr, x_tr).max()
    print(f"step sizes: MSE GD {1 / top:.4f}, logistic GD {4 / top:.4f}, SGD {sgd_gamma:.2e}")
    results = {}

    def curve(name, loss, step, iters):
        """Held-out loss and F1 after each of iters, where step(w, n) runs n more iterations from w. Returns the
        last w."""
        values, w, done, start = [], w0, 0, time()
        for it in iters:
            with np.errstate(over="ignore"):
                w = step(w, it - done)
            done = it
            values.append((loss(y_val, x_val, w), best_f1(y_val, x_val @ w)))
            print(f"{name} step {it}/{iters[-1]}: held-out loss={values[-1][0]:.4f}, F1={values[-1][1]:.4f} "
                  f"({time() - start:.0f}s)")
        values = np.array(values)
        results.update({f"{name}_iters": iters, f"{name}_loss": values[:, 0], f"{name}_f1": values[:, 1]})
        return w

    curve("mse_gd", compute_loss, lambda w, n: mean_squared_error_gd(y_tr, x_tr, w, n, 1 / top)[0], GD_CHECKPOINTS)
    np.random.seed(SEED)  # mean_squared_error_sgd draws its rows from numpy's global generator
    curve("mse_sgd", compute_loss, lambda w, n: mean_squared_error_sgd(y_tr, x_tr, w, n, sgd_gamma)[0],
          np.unique(np.round(np.geomspace(1, len(y_tr), 30)).astype(int)))
    curve("logistic_gd", compute_logistic_loss, lambda w, n: logistic_regression(y_tr, x_tr, w, n, 4 / top)[0],
          GD_CHECKPOINTS)
    curve("reg_logistic_gd", compute_logistic_loss,
          lambda w, n: reg_logistic_regression(y_tr, x_tr, LAMBDA_, w, n, 1 / (top / 4 + 2 * LAMBDA_))[0],
          GD_CHECKPOINTS)
    w = curve("newton", compute_logistic_loss,
              lambda w, n: reg_logistic_regression_newton(y_tr, x_tr, LAMBDA_, w, n)[0], np.arange(1, NEWTON_ITERS + 1))
    # The final model's own decision rule, for comparison with the best-cutoff F1
    results["newton_f1_at_cutoff"] = f1_score(y_val, (sigmoid(x_val @ w) >= THRESHOLD).astype(int))
    return results


def lambda_experiment(x, y):
    """K_FOLD-fold CV (run.py's folds) of ridge regression and of Newton regularized logistic regression for each of
    LAMBDAS, with interaction columns picked on each fold's training rows.

    Returns the best-cutoff F1 on each fold's training and held-out rows, shape (model, lambda, fold), and the
    held-out probabilities of the logistic model at LAMBDA_ for every row (NaN for rows in no fold).
    """
    k_indices = build_k_indices(y, K_FOLD, SEED)
    f1_tr = np.zeros((2, len(LAMBDAS), K_FOLD))
    f1_te = np.zeros_like(f1_tr)
    p_held_out = np.full(len(y), np.nan, dtype=np.float32)
    for k in range(K_FOLD):
        te, tr = k_indices[k], np.delete(k_indices, k, axis=0).ravel()
        x_tr, x_te = x[tr], x[te]
        x_tr, x_te = add_interactions(x_tr, x_te, interaction_pairs(y[tr], x_tr))
        y_tr, y_te = y[tr].astype(x_tr.dtype), y[te]
        w = np.zeros(x_tr.shape[1], dtype=x_tr.dtype)
        # From the largest lambda down, each logistic fit starting from the previous one's weights
        for i in np.argsort(-LAMBDAS):
            start = time()
            w_ridge = ridge_regression(y_tr, x_tr, LAMBDAS[i])[0].astype(x_tr.dtype)
            w, steps = newton_fit(y_tr, x_tr, LAMBDAS[i], w)
            for m, w_m in enumerate((w_ridge, w)):
                f1_tr[m, i, k] = best_f1(y_tr, x_tr @ w_m)
                f1_te[m, i, k] = best_f1(y_te, x_te @ w_m)
            if LAMBDAS[i] == LAMBDA_:
                p_held_out[te] = sigmoid(x_te @ w)
            print(f"fold {k + 1}/{K_FOLD}, lambda={LAMBDAS[i]:.1e}: held-out F1 ridge={f1_te[0, i, k]:.4f}, "
                  f"logistic={f1_te[1, i, k]:.4f} ({steps} Newton steps, {time() - start:.0f}s)")
    return {"lambdas": LAMBDAS, "f1_tr": f1_tr, "f1_te": f1_te, "p_held_out": p_held_out, "y": y}


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_training(r):
    """Held-out MSE (linear models), NLL (logistic models) and F1 (all of them) against the iteration."""
    fig, (ax_mse, ax_nll, ax_f1) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.9))
    for name, (label, color) in CURVES.items():
        style = NEWTON_MARKERS if name == "newton" else {}
        for ax, key in ((ax_mse if name.startswith("mse") else ax_nll, "loss"), (ax_f1, "f1")):
            ax.plot(r[f"{name}_iters"], r[f"{name}_{key}"], color=color, label=label, **style)
    for ax, label in ((ax_mse, "Held-out MSE"), (ax_nll, "Held-out NLL"), (ax_f1, "Held-out F1 (best cutoff)")):
        ax.set(xscale="log", xlabel="Iteration", ylabel=label)
    # Bottom right is empty: by then every curve has risen
    ax_f1.annotate(f"final model at cutoff {THRESHOLD:g}:\nF1 = {float(r['newton_f1_at_cutoff']):.3f}", (0.98, 0.06),
                   xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)
    fig.legend(*ax_f1.get_legend_handles_labels(), loc="outside upper center", ncol=5)
    save(fig, "training.pdf")


def plot_convergence(r):
    """Held-out NLL (left) and F1 (right) against the iteration, GD and Newton on the final model's objective."""
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.75))
    for ax, key, label in zip(axes, ("loss", "f1"), ("Held-out NLL", "Held-out F1 (best cutoff)")):
        ax.plot(r["reg_logistic_gd_iters"], r[f"reg_logistic_gd_{key}"], color=BLUE, label=r"GD, $\gamma = 1/L$")
        ax.plot(r["newton_iters"], r[f"newton_{key}"], color=ORANGE, label="Newton", **NEWTON_MARKERS)
        ax.set(xscale="log", xlabel="Iteration", ylabel=label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    save(fig, "convergence.pdf")


def plot_lambda(r):
    """Mean F1 over the folds against lambda: held-out (solid, with a +-1 std band) and training (dashed)."""
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
    for m, (name, color) in enumerate((("Ridge", BLUE), ("Logistic (Newton)", ORANGE))):
        te = r["f1_te"][m]
        mean, std = te.mean(axis=1), te.std(axis=1)
        ax.fill_between(r["lambdas"], mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(r["lambdas"], mean, color=color, label=name)
        ax.plot(r["lambdas"], r["f1_tr"][m].mean(axis=1), color=color, linewidth=0.8, linestyle="--")
    ax.axvline(LAMBDA_, color=REFERENCE, linewidth=0.6)
    ax.annotate(r"final $\lambda$", (LAMBDA_, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                textcoords="offset points", fontsize=7, color=MUTED)
    ax.set(xscale="log", xlabel=r"Regularization $\lambda$", ylabel="F1 (best cutoff)")
    # Models in the first legend column, line styles in the second
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], color=MUTED, label=r"held-out, $\pm$1 std over folds"),
        Line2D([], [], color=MUTED, linewidth=0.8, linestyle="--", label="training"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)
    save(fig, "lambda.pdf")


def plot_cutoff(r):
    """Held-out precision, recall and F1 of the logistic model at LAMBDA_ against the cutoff on its probability, with
    run.py's cutoff and the default 0.5 marked."""
    valid = ~np.isnan(r["p_held_out"])
    p, y = r["p_held_out"][valid], r["y"][valid]
    precision, recall, f1 = precision_recall_f1(y, p, CUTOFFS)
    marked = np.array([THRESHOLD, 0.5])
    marked_f1 = precision_recall_f1(y, p, marked)[2]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.9))
    ax.plot(CUTOFFS, f1, color=BLUE, label="F1")
    ax.plot(CUTOFFS, precision, color=ORANGE, label="Precision")
    ax.plot(CUTOFFS, recall, color=AQUA, label="Recall")
    for cutoff, value, name in zip(marked, marked_f1, ("final", "default")):
        ax.axvline(cutoff, color=REFERENCE, linewidth=0.6)
        ax.annotate(f"{name} {cutoff:g}\nF1 = {value:.3f}", (cutoff, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -2), textcoords="offset points", va="top", fontsize=7, color=MUTED,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    ax.set(xlim=(0, CUTOFFS[-1]), ylim=(0, 1), xlabel="Cutoff on the predicted probability", ylabel="Held-out score")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3)
    save(fig, "cutoff.pdf")


EXPERIMENTS = {"training": training_experiment, "lambda": lambda_experiment}


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    paths = {name: os.path.join(CACHE_DIR, f"{name}.npz") for name in EXPERIMENTS}
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        now = time()
        print("Loading and cleaning data")
        x, y = load_data()
        print(f"Took {time() - now:.2f} seconds")
        for name in missing:
            now = time()
            print(f"Running the {name} experiment")
            np.savez(paths[name], **EXPERIMENTS[name](x, y))
            print(f"Took {time() - now:.2f} seconds")

    plt.rcParams.update(STYLE)
    results = {name: np.load(path) for name, path in paths.items()}
    plot_training(results["training"])
    plot_convergence(results["training"])
    plot_lambda(results["lambda"])
    plot_cutoff(results["lambda"])


if __name__ == "__main__":
    main()
