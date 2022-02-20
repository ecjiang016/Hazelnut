import numpy as np

class Flatten:
    """
    Reshapes from shape (N, C, H, W) ... stored as (N, C, H*W) in the NN memory ... 
    to shape (C*H*W, N)
    """
    def __init__(self):
        pass
    
    def Forward(self, inp):
        return inp.reshape(inp.shape[0], -1).T

    def Forward_training(self, inp):
        self.N, self.C, self.HW = inp.shape
        return inp.reshape(self.N, -1).T

    def Backward(self, inp):
        return inp.T.reshape(self.N, self.C, self.HW)

    def Build(self):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class Convert4Dto3D:
    """
    Hazelnut's convolution layers take inputs of shape (N, C, H*W).

    This converts CNN inputs from shape (N, C, H, W) to (N, C, H*W) and should be done automatically by Hazelnut
    """
    def __init__(self):
        pass
    
    def Forward(self, inp):
        N, C, _, _ = inp.shape
        return inp.reshape(N, C, -1)

    def Forward_training(self, inp):
        self.N, self.C, self.H, self.W = inp.shape
        return inp.reshape(self.N, self.C, -1)

    def Backward(self, inp):
        return inp.reshape(self.N, self.C, self.H, self.W)

    def Build(self):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass