import numpy as np

def NoOptimizer(learning_rate, weight_gradient, cache=None, hyperparameter=None):
    return weight_gradient * learning_rate, None

def Momentum(learning_rate, weight_gradient, cache=0, hyperparameter=0.9): #Hyperparameter (beta) set at 0.9 or 0.999 for extremely noisy loss funcntion
    momentum_vector = (cache * hyperparameter) + weight_gradient
    return momentum_vector * learning_rate, momentum_vector