import numpy
from . import Optimizers
from ..utils import reshape_to_indices
from ..utils.layer_init.conv_init import He_init

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
        self.optimizer = Optimizers.SGD(0.0001)

        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, inp):
        if self.PAD:
            inp = self.pad(inp)
        
        SN, SC, SH, SW = inp.strides
        col = self.np.lib.stride_tricks.as_strided(
            inp,
            (inp.shape[0], self.OW, self.OH, self.KS, self.KS, self.C), 
            (SN, SW, SH, SW, SH, SC),
            )

        return self.np.swapaxes(self.np.tensordot(col, self.filter, axes=3), 1, 3)

    def Forward_training(self, inp):
        if self.PAD:
            inp = self.pad(inp)

        self.training_cache = inp.copy()
        self.SN, self.SC, self.SH, self.SW = inp.strides
        col = self.np.lib.stride_tricks.as_strided(
            inp,
            (inp.shape[0], self.OW, self.OH, self.KS, self.KS, self.C), 
            (self.SN, self.SW, self.SH, self.SW, self.SH, self.SC),
            )

        return self.np.swapaxes(self.np.tensordot(col, self.filter, axes=3), 1, 3)

    def Backward(self, inp):
        #Calculate the next gradient
        padded_filter = self.pad_filter(self.np.flip(self.filter, axis=(0, 1)))
        SW, SH, SC, SF = padded_filter.strides

        filter_col = self.np.lib.stride_tricks.as_strided(
            padded_filter,
            (self.W, self.H, self.C, self.OW, self.OH, self.F), 
            (SW, SH, SC, SW, SH, SF),
            )

        #inp reshape: (N, F, OH, OW) -> (OW, OH, F, N)
        pass_gradient = self.np.tensordot(filter_col, inp.transpose(3, 2, 1, 0), axes=3).transpose(3, 2, 1, 0) # (W, H, C, N) -> (N, C, H, W)

        #Calculate filter gradient and update filter
        cache_acti_col = self.np.lib.stride_tricks.as_strided(
            self.training_cache,
            (self.KS, self.KS, self.C, self.OW, self.OH, inp.shape[0]), 
            (self.SW, self.SH, self.SC, self.SW, self.SH, self.SN),
            )
            
        #inp reshape: (N, F, OH, OW) -> (OW, OH, N, F)
        filter_grad = self.np.tensordot(cache_acti_col, inp.transpose(3, 2, 0, 1), axes=3)
        self.filter -= self.optimizer.use(filter_grad)
        
        if self.PAD:
            return pass_gradient[:, :, self.PAD:-self.PAD, self.PAD:-self.PAD]

        
        return pass_gradient
         
    def pad(self, inp):
        padded_activations = self.np.zeros((inp.shape[0], self.C, self.H, self.W))
        padded_activations[:, :, self.PAD_SIZE:-self.PAD_SIZE, self.PAD_SIZE:-self.PAD_SIZE] = inp
        return padded_activations

    def pad_filter(self, filter_):
        padded_filter = self.np.zeros((self.PFW, self.PFH, self.C, self.F))
        padded_filter[self.PAD_F_SIZE_W:-self.PAD_F_SIZE_W, self.PAD_F_SIZE_H:-self.PAD_F_SIZE_H, :, :] = filter_
        return padded_filter
    
    def Build(self, shape):
        _, self.C, self.H, self.W = shape

        #Initialize filters
        self.filter = self.np.array(self.init_method(self.F, self.C, self.KS))
        self.filter = self.filter.transpose(3, 2, 1, 0) #Swapped to KS[W], KS[H], C, F

        #Padding calculations
        if self.mode == 'Same':
            self.PAD_SIZE = (self.KS - 1) // 2
            #Zero padding matrix 
            self.H += self.KS - 1
            self.W += self.KS - 1

        #Calculate output dimensions
        self.OH = self.H - self.KS + 1
        self.OW = self.W - self.KS + 1

        #Padding calculating for full convolution
        self.PAD_F_SIZE_H = self.OH - 1
        self.PAD_F_SIZE_W = self.OW - 1
        self.PFH = self.KS + self.PAD_F_SIZE_H + self.PAD_F_SIZE_H
        self.PFW = self.KS + self.PAD_F_SIZE_W + self.PAD_F_SIZE_W

    def Save(self):
        return {'args':(self.F, self.KS, self.mode, self.init_method), 'var':(self.filter, self.optimizer.__class__, self.optimizer.Save())}

    def Load(self, var):
        self.filter, optimizer_class, optimizer_dict = var
        self.optimizer = optimizer_class()
        self.optimizer.load(optimizer_dict)
