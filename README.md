# EPFL CS-433 ML Project 1

## Introduction

This repository contains the code for our first project submission for the EPFL Machine Learning Course CS-433.

## Requirements

The only requirements as also stipulated by the project description, is NumPy. We were prohibited to use any external machine learning (ML) libraries. Numpy can easily be installed with the command: `pip install numpy`. We recommend doing so in a virtual environment using venv or conda.

## File Descriptions

The following section describes the functionality and contents of each of the Python scripts that we have in this repository.

### `helpers.py`

This Python file contains helper functions (as the name indicates) that load and store CSV files. This is necessary because the dataset we are dealing with is in CSV format and we are not allowed to use packages like Pandas to load in datasets as Dataframes.

### `implementations.py`

This Python file contains the six basic method implementations: `mean squared error gd`, `mean squared error sgd`, `least squares`, `ridge regression`, `logistic regression`, and `reg logistic regression`. Additionally, we added implementations for k-fold cross-validation, logistic regression trained using Newton's method, and also helper functions like computing F1-scores.

### `cleaner.py`

This Python file contains most of our work as it pertains to cleaning and parsing the dataset. We went through several iterations of what columns to keep, how to encode them, and what additional data we may be able to assist.

### `main.py`

This Python file is/was responsible for actually running the the different stages of the pipeline and is the file we used to find the hyperparameters and best model for the given dataset that we could determine.

### `run.py`

This Python file trains our best model `reg_logistic_regression_newton` using the hyperparameters we found in `main.py`. Since this process may take a while, we load a pickle file that froze our trained weights. To train the model yourself, you simply need to uncomment the appropriate lines in the `run.py` file as indicated.