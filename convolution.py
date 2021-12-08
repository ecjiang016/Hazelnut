import numpy as np
from scipy.signal import convolve as Conv
from .ActivationFunctions import *

class convolution:
    def __init__(self, Layout):
        self.Layout = Layout
    
    def Pool(self, matrix, size):
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
        history = [("Input", input_array)]
        for module in self.Layout:
            module_type = module[0]
            activations = history[-1][1]

            if module_type == "Kernel":
                output = Conv(activations, module[1], "valid")
                history.append((module_type, output))
            
            elif module_type == "Pooling":
                output, mask = self.Pool(activations, module[1])
                history.append((module_type, output, mask))
                
            elif module_type == "Acti_func":
                output = AF(activations, module[1])
                history.append((module_type, output))
            
            else:
                raise ValueError("Invalid convolution module layout")
        
        return history
    
    def Backpropagate(self, History, PreviousGradient, learning_rate):
        kernel_gradients = {}
        for i, module in enumerate(self.Layout):
            if module[0] == "Kernel":
                kernel_gradients[i] = 0

        for example_num, previous_gradient in enumerate(PreviousGradient):
            history = History[example_num]
            for j, module in enumerate(reversed(self.Layout)):
                i = 0-j-1
                module_type = module[0]

                if module_type == "Kernel":
                    flipped_kernel = np.flipud(np.fliplr(module[1]))
                    index = len(self.Layout)+i
                    kernel_gradients[index] += Conv(history[i-1][1], previous_gradient, "valid")
                    previous_gradient = Conv(flipped_kernel, previous_gradient, "full")
                    
                elif module_type == "Pooling":
                    previous_gradient = self.Pool_Backpropagate(previous_gradient, history[i][2], history[i-1][1].shape, module[1])

                elif module_type == "Acti_func":
                    previous_gradient = previous_gradient * AF_dv(history[i][1], module[1])

                else:
                    raise ValueError("Invalid convolution module layout")
        
        for index, gradient in kernel_gradients.items():
            self.Layout[index] = ("Kernel", self.Layout[index][1] + ((gradient/len(PreviousGradient)) * learning_rate))


    def Backpropagate_single(self, history, PreviousGradient, learning_rate):
        previous_gradient = PreviousGradient
        
        for j, module in enumerate(reversed(self.Layout)):
            i = 0-j-1
            module_type = module[0]

            if module_type == "Kernel":
                flipped_kernel = np.flipud(np.fliplr(module[1]))
                gradient = Conv(history[i-1][1], previous_gradient, "valid")
                self.Layout[i] = ("Kernel", module[1] - (gradient * learning_rate))
                previous_gradient = Conv(flipped_kernel, previous_gradient, "full")
                
            elif module_type == "Pooling":
                previous_gradient = self.Pool_Backpropagate(previous_gradient, history[i][2], history[i-1][1].shape, module[1])

            elif module_type == "Acti_func":
                previous_gradient = previous_gradient * AF_dv(history[i][1], module[1])

            else:
                raise ValueError("Invalid convolution module layout")