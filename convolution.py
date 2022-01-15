import numpy as np
from scipy.signal import convolve
from .ActivationFunctions import *
from .MatmulConv.Conv import conv, conv_full
from alive_progress import alive_it as bar
import Optimizers

class convolution:
    def __init__(self, Layout):
        self.Layout = Layout #Layout = [("Filter", [filter_0, filter_1, filter_2]], "Valid"), ("AF", "ReLU", bias), ("Pooling", size)]
        self.LearningRate = 0.0001
        self.Optimizer = Optimizers.Momentum

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
        for module in self.Layout:
            module_type = module[0]
            activations = history[-1]

            if module_type == "Filter":
                if module[2] == "Same":
                    activations = np.pad(activations, (module[1].shape - 1) //2)
                history.append(conv(activations, module[1]))
            
            elif module_type == "Pooling":
                output, mask = self.Pool(activations, module[1])
                history.append((output, mask))
                
            elif module_type == "AF":
                #output = AF(activations + module[2], module[1])
                output = AF(activations, module[1])
                history.append(output)
            
            else:
                raise ValueError("Invalid convolution module layout")
        
        return history
    
    def Backpropagate(self, history, previous_gradient):
        batch_size = previous_gradient.shape[0]
        last = len(self.Layout) -1 #Reduces amount of len() calls

        for index, module in enumerate(reversed(self.Layout)): #Change to reversed(enumerate(self.Layout)) sometime

            i = last - index #Compromises for the reversing of the layout but not the index
            module_type = module[0]

            if module_type == "Filter":
                acti = np.swapaxes(history[i], 0, 1)
                #Calculate and update kernel gradients
                grad = np.swapaxes(previous_gradient, 0, 1)
                kernel_gradients = np.swapaxes(conv(acti, grad), 0, 1)/batch_size

                if module[2] == "Same":
                    padding_size = (module[1].shape -1) //2
                    kernel_gradients = kernel_gradients[:, :, padding_size:-padding_size, padding_size:-padding_size]
                    previous_gradient = np.pad(previous_gradient, padding_size)

                kernel_gradients, self.OptimizerCache[i] = self.Optimizer(self.LearningRate, kernel_gradients, self.OptimizerCache[i])
                self.Layout[i] = ("Filter", self.Layout[i][1] - kernel_gradients)

                #Calculate the next gradient
                new_kern = np.flip(np.swapaxes(module[1], 0, 1), (2, 3))
                previous_gradient = conv_full(previous_gradient, new_kern)
    
            elif module_type == "Pooling":
                previous_gradient = self.Pool_Backpropagate(previous_gradient, history[i+1][1], history[i][0].shape, module[1])

            elif module_type == "AF":
                previous_gradient = previous_gradient * AF_dv(history[i], module[1])

            else:
                raise ValueError("Invalid convolution module layout")
