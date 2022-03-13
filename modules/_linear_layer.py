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
        self.training_cache = inp
        return self.np.matmul(self.weights, inp) + self.biases[:, None]

    def Backward(self, inp):
        self.weights -= self.optimizer_weights.use(self.np.matmul(inp, self.training_cache.T) / inp.shape[0])
        self.biases -= self.optimizer_biases.use(self.np.sum(inp, axis=1) / inp.shape[0])
        return self.np.matmul(self.weights.T, inp)

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
        return {'args':(self.Neurons, self.init_method), 'var':(self.weights, self.biases, self.optimizer.__class__, self.optimizer.Save())}

    def Load(self, var):
        self.weights, self.biases, optimizer_class, optimizer_dict = var
        self.optimizer = optimizer_class()
        self.optimizer.load(optimizer_dict)