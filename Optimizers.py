import numpy as np

def NoOptimizer(learning_rate, weight_gradient, cache=None):
    return weight_gradient * learning_rate

def Momentum(learning_rate, weight_gradient, cache=0):
    beta = 0.9 #Hyperparameter set at 0.9 or 0.999 for extremely noisy loss funcntion
    return ((cache * beta) + weight_gradient) * learning_rate