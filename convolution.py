import numpy as np
from scipy.signal import convolve
from .ActivationFunctions import *
from .MatmulConv.Conv import conv, conv_full
from . import Optimizers
from . import BatchNormalization as BatchNorm

class convolution:
    def __init__(self, Layout, Training=False):
        self.Layout = Layout #Layout = [("Skip", "Begin"), ("Filter", [filter_0, filter_1, filter_2]], "Valid"), ("BatchNorm", (gamma=1, beta=0, test_mean=0, test_variance=0)), ("AF", "ReLU",), ("Skip", "End")]
        self.LearningRate = 0.0001
        self.Optimizer = Optimizers.Momentum
        self.Training = Training

        #Global variables handled by the class (Don't mess with them)
        self.OptimizerCache = {}
        for i, module in enumerate(self.Layout):
            if module[0] == "Filter":
                self.OptimizerCache[i] = 0
    
    def Pool(self, matrix, size): #Does not currently work due to the lack of tensor support
        strides_y = int(np.floor(matrix.shape[0]/size))
        strides_x = int(np.floor(matrix.shape[1]/size))
        mask = np.zeros(matrix.shape)
        output = np.zeros((strides_y, strides_x))
        for y in range(0, strides_y):
            for x in range(0, strides_x):
                slice = matrix[y*size:y*size+size, x*size:x*size+size]
                max_val = np.max(slice)
                output[y][x] = max_val
                mask_y, mask_x = np.where(slice == max_val)
                mask_y, mask_x = mask_y[0], mask_x[0]
                mask[y*size+mask_y][x*size+mask_x] = 1
        return output, mask
        
    def Pool_Backpropagate(self, previous_gradient, mask, previous_matrix_size, size):
        strides_y = int(np.floor(previous_matrix_size[0]/size))
        strides_x = int(np.floor(previous_matrix_size[1]/size))
        
        output = np.zeros(previous_matrix_size)
        
        for y in range(0, strides_y):
            for x in range(0, strides_x):
                output[y*size:y*size+size, x*size:x*size+size] = previous_gradient[y, x] * mask[y*size:y*size+size, x*size:x*size+size]

        return output

    def FeedForward(self, input_array):
        history = [input_array]
        activations = input_array
        for module in self.Layout:
            module_type = module[0]

            if module_type == "Filter":
                if module[2] == "Same":
                    activations = np.pad(activations, (module[1].shape - 1) //2)
                history.append(conv(activations, module[1]))

            elif module_type == "Batch Norm":
                activations, cache = BatchNorm.FeedForward(activations, module[1], training=self.Training)
                history.append(cache)
                
            elif module_type == "AF":
                activations = AF(activations, module[1])
                history.append(activations)

            elif module_type == "Skip":
                if module[1] == "Begin":
                    skip_cache = activations
                else:
                    activations += skip_cache

            elif module_type == "Pooling":
                output, mask = self.Pool(activations, module[1])
                activations = output
                history.append((output, mask))
            
            else:
                raise ValueError("Invalid convolution module layout")
        
        return history
    
    def Backpropagate(self, history, pass_gradient):
        batch_size = pass_gradient.shape[0]
        last = len(self.Layout) -1 #Reduces amount of len() calls

        for index, module in enumerate(reversed(self.Layout)): #Change to reversed(enumerate(self.Layout)) sometime

            i = last - index #Compromises for the reversing of the layout but not the index
            module_type = module[0]

            if module_type == "Filter":
                acti = np.swapaxes(history[i], 0, 1)
                #Calculate and update kernel gradients
                grad = np.swapaxes(pass_gradient, 0, 1)
                kernel_gradients = np.swapaxes(conv(acti, grad), 0, 1)/batch_size

                if module[2] == "Same":
                    padding_size = (module[1].shape -1) //2
                    kernel_gradients = kernel_gradients[:, :, padding_size:-padding_size, padding_size:-padding_size]
                    pass_gradient = np.pad(pass_gradient, padding_size)

                kernel_gradients, self.OptimizerCache[i] = self.Optimizer(self.LearningRate, kernel_gradients, self.OptimizerCache[i])
                self.Layout[i] = ("Filter", self.Layout[i][1] - kernel_gradients)

                #Calculate the next gradient
                new_kern = np.flip(np.swapaxes(module[1], 0, 1), (2, 3))
                pass_gradient = conv_full(pass_gradient, new_kern)

            elif module_type == "BatchNorm":
                pass_gradient, module[1] = ("BatchNorm", BatchNorm.Backpropagate(pass_gradient, module[1], history[i], self.LearningRate))

            elif module_type == "AF":
                pass_gradient = pass_gradient * AF_dv(history[i], module[1])

            elif module_type == "Skip":
                if module[1] == "End":
                    skip_cache = pass_gradient
                else:
                    pass_gradient += skip_cache


            elif module_type == "Pooling":
                pass_gradient = self.Pool_Backpropagate(pass_gradient, history[i+1][1], history[i][0].shape, module[1])

            else:
                raise ValueError("Invalid convolution module layout")
