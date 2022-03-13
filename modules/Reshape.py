class Flatten:
    """
    Reshapes from shape (N, C, H, W) to shape (C*H*W, N)
    """
    def __init__(self):
        pass
    
    def Forward(self, inp):
        return inp.reshape(inp.shape[0], -1).T

    def Forward_training(self, inp):
        self.N, self.C, self.H, self.W = inp.shape
        return inp.reshape(self.N, -1).T

    def Backward(self, inp):
        return inp.T.reshape(self.N, self.C, self.H, self.W)

    def Build(self, _):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass
