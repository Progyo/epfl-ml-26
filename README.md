# EPFL CS-433 ML Project 1

## Introduction

This repository contains the code for our first project submission for the EPFL Machine Learning Course CS-433.

## Requirements

The only requirement, as also stipulated by the project description, is NumPy. We were prohibited to use any external machine learning (ML) libraries. NumPy can easily be installed with the command: `pip install numpy`. We recommend doing so in a virtual environment using venv or conda. Note that for the `plotter.py` script, `matplotlib 3.7` or newer is required.

## Dataset

The dataset must be placed in a folder called `dataset` in the root of this repository. If you need to change this, you can do so in the according script that you would like to run.

## File Descriptions

The following section describes the functionality and contents of each of the Python scripts that we have in this repository.

### `helpers.py`

This Python file contains helper functions (as the name indicates) that load and store CSV files. This is necessary because the dataset we are dealing with is in CSV format and we are not allowed to use packages like Pandas to load in datasets as Dataframes.

### `implementations.py`

This Python file contains the six basic method implementations: `mean squared error gd`, `mean squared error sgd`, `least squares`, `ridge regression`, `logistic regression`, and `reg logistic regression`. Additionally, we added implementations for k-fold cross-validation, logistic regression trained using Newton's method, and also helper functions like computing F1-scores.

### `cleaner.py`

This Python file contains most of our work as it pertains to cleaning and parsing the dataset. We went through several iterations of what columns to keep, how to encode them, and what additional data we may be able to assist.

### `old_cleaner.py`

This is an old version of our cleaner.py file that we used a baseline for our ablation study. It drops columns that are at least half missing, keeps categorical answers as ordinal ranks instead of one-hot encoding them, fills missing values without a missing indicator, and adds no interaction columns.

### `main.py`

This Python file is/was responsible for actually running the the different stages of the pipeline and is the file we used to find the hyperparameters and best model for the given dataset that we could determine. `python main.py` compares all models on the current cleaning; `python main.py --old-cleaner` runs the same comparison on `old_cleaner.py` and writes `submission_old_cleaner.csv` instead of `submission.csv`.

### `ablation.py`

The ablation study in the report. It trains our final model with the hyperparameters of `run.py` on the same 80/20 split as `main.py`, with steps of the pipeline switched off by flags: `--no-interactions`, `--no-codebook`, `--drop-half-missing`, `--no-one-hot`, `--no-missing-indicators` and `--old-cleaner`, which can be combined. `python ablation.py --all` runs the final model, each step off on its own, and the old cleaning in one go, and also scores the final model at the default cutoff 0.5.

### `run.py`

This Python file trains our best model `reg_logistic_regression_newton` using the hyperparameters we found in `main.py`. Since this process may take a while, we load our trained weights from `model_weights.npy`. To train the model yourself, you simply need to uncomment the appropriate lines in the `run.py` file as indicated.