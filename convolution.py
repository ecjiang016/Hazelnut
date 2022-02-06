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
        self.OptimizerHyperparameter = 0.9
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
        history = []
        activations = input_array
        for module in self.Layout:
            module_type = module[0]

            if module_type == "Filter":
                history.append(activations)

                if module[2] == "Same":
                    N, C, H, W = activations.shape
                    padded_activations = np.zeros((N, C, H+module[1].shape[2] - 1, W+module[1].shape[2] - 1))

                    pad_size = (module[1].shape[2] -1) // 2
                    padded_activations[:, :, pad_size:-pad_size, pad_size:-pad_size] = activations
                    activations = padded_activations

                activations = conv(activations, module[1])


            elif module_type == "Batch Norm":
                activations, cache = BatchNorm.FeedForward(activations, module[1], training=self.Training)
                history.append(cache)
                
            elif module_type == "AF":
                history.append(activations)
                activations = AF(activations, module[1])

            elif module_type == "Skip":
                if module[1] == "Begin":
                    skip_cache = activations
                else:
                    activations += skip_cache

                history.append(None)

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
                acti = history[i]
                if module[2] == "Same":
                    N, C, H, W = acti.shape
                    padded_acti = np.zeros((N, C, H+module[1].shape[2] - 1, W+module[1].shape[2] - 1))

                    pad_size = (module[1].shape[2] -1) // 2
                    padded_acti[:, :, pad_size:-pad_size, pad_size:-pad_size] = acti

                    acti = padded_acti

                acti = np.swapaxes(acti, 0, 1)
                #Calculate and update kernel gradients
                grad = np.swapaxes(pass_gradient, 0, 1)
                kernel_gradients = np.swapaxes(conv(acti, grad), 0, 1)/batch_size

                #Calculate the next gradient
                new_kern = np.flip(np.swapaxes(module[1], 0, 1), (2, 3))
                pass_gradient = conv_full(new_kern, pass_gradient)
                pass_gradient = np.swapaxes(pass_gradient, 0, 1)

                if module[2] == "Same":
                    padding_size = (module[1].shape[2] -1) //2
                    pass_gradient = pass_gradient[:, :, padding_size:-padding_size, padding_size:-padding_size]

                kernel_gradients, self.OptimizerCache[i] = self.Optimizer(self.LearningRate, kernel_gradients, self.OptimizerCache[i], self.OptimizerHyperparameter)
                self.Layout[i] = ("Filter", self.Layout[i][1] - kernel_gradients, module[2])

            elif module_type == "Batch Norm":
                pass_gradient, parameters = BatchNorm.Backpropagate(pass_gradient, module[1], history[i], self.LearningRate)
                module = ("Batch Norm", parameters)

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