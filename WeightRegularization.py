import numpy as np

def NoRegularization(weights, hyperparameter):
    return 0

def L2(weights, hyperparameter):
    return hyperparameter * np.square(weights) / weights.size