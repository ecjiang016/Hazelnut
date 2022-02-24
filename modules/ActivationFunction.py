import numpy as np

class ReLU:
    def __init__(self):
        pass
    
    def Forward(self, inp):
        inp[inp < 0] = 0
        return inp

    def Forward_training(self, inp):
        self.training_cache = inp.copy()
        inp[inp < 0] = 0
        return inp

    def Backward(self, inp):
        return (self.training_cache >= 0) * inp

    def Build(self, _):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class tanh:
    def __init__(self):
        pass
    
    def Forward(self, inp):
        return np.tanh(inp)

    def Forward_training(self, inp):
        self.training_cache = inp.copy()
        return np.tanh(inp)

    def Backward(self, inp):
        return (1 - np.square(np.tanh(self.training_cache))) * inp

    def Build(self, _):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class Sigmoid:
    def __init__(self):
        pass
    
    def Forward(self, inp):
        return 1/(np.exp(-inp)+1)

    def Forward_training(self, inp):
        out = 1/(np.exp(-inp)+1)
        self.training_cache = out.copy()
        return out

    def Backward(self, inp):
        return (self.training_cache * (1 - self.training_cache)) * inp

    def Build(self, _):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass
