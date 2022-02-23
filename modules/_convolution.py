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
        
        SN, SC, SH, SW = inp.strides
        col = as_strided(
            inp,
            (inp.shape[0], self.OW, self.OH, self.KS, self.KS, self.C), 
            (SN, SW, SH, SW, SH, SC),
            writeable=False)

        return np.swapaxes(np.tensordot(col, self.filter, axes=3), 1, 3)

    def Forward_training(self, inp):
        if self.PAD:
            inp = self.pad(inp)

        self.training_cache = inp.copy()
        SN, SC, SH, SW = inp.strides
        self.acti_strides = (SN, SW, SH, SW, SH, SC)
        col = as_strided(
            inp,
            (inp.shape[0], self.OW, self.OH, self.KS, self.KS, self.C), 
            self.acti_strides,
            writeable=False)

        return np.swapaxes(np.tensordot(col, self.filter, axes=3), 1, 3)

    def Backward(self, inp):
        reshaped_inp = np.swapaxes(np.swapaxes(inp, 0, 3), 1, 2)

        #Calculate the next gradient
        SW, SH, SC, SF = self.filter.strides
        filter_col = as_strided(
            np.flip(self.filter, axis=(0, 1)),
            (self.N, self.W, self.H, self.OW, self.OH, self.F), 
            (SF, SW, SH, SW, SH, SC),
            writeable=False)

        pass_gradient = np.swapaxes(np.tensordot(filter_col, reshaped_inp, axes=3), 1, 3)

        #Calculate filter gradient and update filter
        cache_acti_col = as_strided(
            inp,
            (inp.shape[0], self.KS, self.KS, self.OW, self.OH, self.F), 
            self.acti_strides,
            writeable=False)
            
        filter_grad = np.swapaxes(np.tensordot(cache_acti_col, reshaped_inp, axes=3), 1, 3)
        self.filter = self.optimizer.use(np.sum(filter_grad, axis=0))
        
        if self.PAD:
            return pass_gradient[self.unpadding_indices]
        
        return pass_gradient
         
    def pad(self, inp):
        padded_activations = np.zeros((inp.shape[0], self.C, self.H, self.W))
        padded_activations[:, :, self.PAD_SIZE:-self.PAD_SIZE, self.PAD_SIZE:-self.PAD_SIZE] = inp
        return padded_activations

    def pad_filter(self, inp):
        padded_filter = np.zeros((inp.shape[0], self.C, self.PFH, self.PDW))
        padded_filter[:, :, self.PAD_F_SIZE:-self.PAD_F_SIZE, self.PAD_F_SIZE:-self.PAD_F_SIZE] = inp
        return padded_filter
    
    def Build(self, inp):
        _, self.C, self.H, self.W = inp.shape

        #Initialize filters
        self.filter = self.init_method(self.F, self.C, self.KS)
        self.filter = np.swapaxes(np.swapaxes(self.filter, 0, 3), 1, 2)

        #Calculate output dimensions
        self.OH = self.H - self.KS + 1
        self.OW = self.W - self.KS + 1

        #Padding calculations
        if self.mode == 'Same':
            self.PAD_SIZE = (self.KS - 1) // 2
            #Zero padding matrix 
            self.H += self.KS - 1
            self.W += self.KS - 1
            #Fix output dimensions
            self.OH = self.H
            self.OW = self.W

        #Padding calculating for full convolution
        self.PAD_F_SIZE = self.KS - 1
        self.PFH = self.H + self.KS + self.KS - 2
        self.PDW = self.W + self.KS + self.KS - 2

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
