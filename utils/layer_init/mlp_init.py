import numpy as np

#Note: Keep these as functions that return numpy arrays and convert to cupy in the modules if needed

def Xavior_init(neurons_num, prev_neurons_num): #For sigmoid and tanh
    weights = np.random.normal(size=(neurons_num, prev_neurons_num))*np.sqrt(1/neurons_num)
    biases = np.zeros(neurons_num)
    return weights, biases

def He_init(neurons_num, prev_neurons_num): #For ReLU and Leaky ReLU
    weights = np.random.normal(size=(neurons_num, prev_neurons_num))*np.sqrt(2/neurons_num)
    biases = np.full(neurons_num, 0.1)
    return weights, biases