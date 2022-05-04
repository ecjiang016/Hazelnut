import numpy
from . import Optimizers
from ..utils.layer_init.mlp_init import He_init

class Linear:
    def __init__(self, Neurons:int, init_method=He_init):
        """
        Args:
        - F: Number of filters
        - KS: Kernel size
        - mode: 'Valid' or 'Same'. The method used for the convolution
        - init_method: A function with args (neurons_num, prev_neurons_num).
        With neurons_num = number of neurons in the layer
        and prev_neurons_num = number of neurons in the previous linear layer and/or input length.
        Outputs the weight and biases as a tuple with weights in shape (neurons_num, prev_neurons_num)
        and biases in shape (neurons_num,)
        """

        self.Neurons = Neurons

        self.init_method = init_method

        #Defaults
        self.optimizer = Optimizers.SGD(0.0001)

        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, inp):
        return self.np.matmul(self.weights, inp) + self.biases[:, None]

    def Forward_training(self, inp):
        self.training_cache = inp.copy()
        return self.np.matmul(self.weights, inp) + self.biases[:, None]

    def Backward(self, inp):
        weight_gradient = self.np.matmul(inp, self.training_cache.T) / inp.shape[1]
        bias_gradient = self.np.sum(inp, axis=1) / inp.shape[1]
        self.gradient = (weight_gradient, bias_gradient)
        return self.np.matmul(self.weights.T, inp)

    def Update(self):
        weight_grad, bias_grad = self.gradient
        #raise Exception
        self.weights -= self.optimizer_weights.use(weight_grad)
        self.biases -= self.optimizer_biases.use(bias_grad)

    def Build(self, shape):
        self.inp_size = shape[0]

        #Make 2 different optimizers for weights and biases
        from copy import deepcopy
        optimizer_np = self.optimizer.np
        self.optimizer.np = None #Can't have this be a module (NumPy or CuPy) as deepcopying won't work correctly

        self.optimizer_weights = deepcopy(self.optimizer)
        self.optimizer_biases = deepcopy(self.optimizer)
        #Put the removed np back
        self.optimizer_weights.np = optimizer_np
        self.optimizer_biases.np = optimizer_np

        self.weights, self.biases = self.init_method(self.Neurons, self.inp_size)

        #Hazelnut's layer init functions return NumPy ndarrays.
        #If using CuPy instead of NumPy, the arrays will be converted to CuPy arrays.
        self.weights = self.np.array(self.weights)
        self.biases = self.np.array(self.biases) 

    def Save(self):
        return {'args':(self.Neurons,), 'var':(self.weights, self.biases)}

    def Load(self, var):
        self.weights, self.biases = var