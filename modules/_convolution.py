import numpy as np
from numpy.lib.stride_tricks import as_strided
from . import Optimizers
from ..utils import reshape_to_indices
from ..utils.layer_init import He_init

class Conv:
    def __init__(self, F, KS, mode='Valid', init_method=He_init):
        """
        Args:
        - F: Number of filters
        - KS: Kernel size
        - mode: 'Valid' or 'Same'. The method used for the convolution
        - init_method: A function with args (F, C, KS). F = number of filters, C = number of channels, KS = kernel/filer size
        and outputs a filter with size (F, C, KS, KS).
        """
        self.F = F
        self.KS = KS

        if mode not in ['Valid', 'Same']:
            raise ValueError("Incorrect Conv mode")
        self.mode = mode
        if self.mode == 'Same':
            self.PAD = True
        else:
            self.PAD = False

        self.init_method = init_method

        #Defaults
        self.optimizer = Optimizers.NoOptimizer(0.0001)

    def Forward(self, inp):
        if self.PAD:
            inp = self.pad(inp)
        
        return np.matmul(self.filter, inp[:, self.channel_indices, self.width_indices])

    def Forward_training(self, inp):
        if self.PAD:
            inp = self.pad(inp)

        self.training_cache = inp.copy()
        
        return np.matmul(self.filter, inp[:, self.channel_indices, self.width_indices])

    def Backward(self, inp):
        #Calculate filter gradient and update filter
        reshaped_inp = np.reshape(np.flip(np.swapaxes(inp, 0, 1), 2), (self.C, -1))
        self.filter -= self.optimizer.use(np.swapaxes(np.matmul(reshaped_inp.view(), np.swapaxes(self.training_cache, 0, 1)[:, self.channel_indices, self.width_indices]), 0, 1) / inp.shape[0])
        
        #Calculate the next gradient
        reshaped_inp2 = np.reshape(np.flip(inp, 2), (inp.shape[0], -1))
        pass_gradient = np.swapaxes(np.matmul(reshaped_inp2.view(), self.filter[self.filter_reshape_indices]), 0, 1)

        if self.PAD:
            return pass_gradient[self.unpadding_indices]
        
        return pass_gradient
         

    def pad(self, inp):
        padded_activations = np.zeros((inp.shape[0], self.C, self.H+self.KS - 1, self.W+self.KS - 1))
        padded_activations[:, :, self.PAD_SIZE:-self.PAD_SIZE, self.PAD_SIZE:-self.PAD_SIZE] = inp
        return padded_activations

    def pad_filter(self, inp):
        padded_filter = np.zeros((inp.shape[0], self.C, self.H+self.KS+self.KS-2, self.W+self.KS+self.KS-2))
        padded_filter[:, :, self.PAD_F_SIZE:-self.PAD_F_SIZE, self.PAD_F_SIZE:-self.PAD_F_SIZE] = inp
        return padded_filter
    
    def Build(self, inp):
        _, self.C, self.H, self.W = inp.shape

        #Initialize filters
        self.filter = self.init_method(self.F, self.C, self.KS)
        self.filter = np.reshape(np.flip(self.filter, (2, 3)), (self.F, -1))

        #Calculate output dimensions
        self.OH = self.H - self.KS + 1
        self.OW = self.W - self.KS + 1

        #Padding calculations
        if self.mode == 'Same':
            self.PAD_SIZE = (self.KS - 1) // 2
            self.H += self.KS - 1
            self.W += self.KS - 1
        
        self.PAD_F_SIZE = self.KS - 1

        #Generate indices for feed forward
        
        indices = np.arange(self.C*self.H*self.W).reshape(self.C, self.H, self.W)
        #Use im2col on it
        SC, SH, SW = indices.strides
        indices = as_strided(indices,
                            shape=(self.C, self.KS, self.KS, self.OH, self.OW),
                            strides=(SC, SH, SW, SH, SW),
                            ).reshape((self.C*self.KS*self.KS, self.OH*self.OW))
        #Some magic happens here
        self.channel_indices = indices // (self.H * self.W) % self.C
        self.width_indices = indices % (self.H * self.W)


        #Generate indices for the pass gradient convolution
        @reshape_to_indices
        def filter_im2col(inp):
            reshaped_inp = inp.reshape(self.F, self.C, self.KS, self.KS)
            SF, SC, SH, SW = reshaped_inp.strides
            out = as_strided(reshaped_inp,
                                shape=(self.C, self.KS, self.KS, self.F, self.OH, self.OW),
                                strides=(SC, SH, SW, SF, SH, SW),
                                ).reshape((self.C*self.KS*self.KS, self.F*self.OH*self.OW))

            return out

        self.filter_reshape_indices = filter_im2col(self.filter)


        #Generate indices for "unpadding" the pass gradient for a same mode convultion layer
        if self.PAD:
            @reshape_to_indices
            def unpad(inp):
                reshaped_inp = inp.reshape(self.H+(2*self.PAD_SIZE), self.W+(2*self.PAD_SIZE))
                reshaped_inp = reshaped_inp[self.PAD_SIZE:-self.PAD_SIZE, self.PAD_SIZE:-self.PAD_SIZE]

            self.unpadding_indices = unpad(np.zeros(self.H+(2*self.PAD_SIZE), self.W+(2*self.PAD_SIZE)))
            self.unpadding_indices.insert(..., 0)
            self.unpadding_indices.insert(..., 0)
            self.unpadding_indices = tuple(self.unpadding_indices)

    def Save(self):
        return {'args':(self.F, self.KS, self.mode, self.init_method), 'var':(self.filter, self.optimizer)}

    def Load(self, var):
        self.filter, self.optimizer = var


if __name__ == '__main__':
    conv = Conv(32, 3, mode='Valid')
    conv.Precache(np.zeros((1, 32, 8, 8)))

    inp = np.arange(24).reshape(2, 3, 2, 2)
    #out = conv.Forward_training(inp)
    
    #inps = np.swapaxes(inp, 0, 1)[:, conv.channel_indices, conv.width_indices]

    #a = np.reshape(np.flip(np.swapaxes(inp, 0, 1), 2), (2, -1))
    #print(a)

    #b = np.reshape(np.flip(inp, 2), (2, -1))
    #print(b)
