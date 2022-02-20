import numpy as np
from . import Optimizers

class BatchNorm:
    """
    - epsilon: Constant that helps with stability by preventing division by zero
    - 
    """
    def __init__(self) -> None:
        self.decay_rate = 0.9 #Constant determining how important the latest mean and variance is in the mean
        self.epsilon = 0.0001 #Constant that helps with stability by preventing division by zero

        #Parameters to scale (gamma) and shift (beta) the normalized activations. These are learned by the net
        self.gamma = 1
        self.beta = 0 

        #Mean and variance for when batch norm is not used in training
        self.test_mean = 0 
        self.test_variance = 1

        #Default optimizer
        self.optimizer = Optimizers.NoOptimizer(0.0001)

    def Forward(self, inp):
        return (inp - self.test_mean[None, :, None]) * (1 / np.sqrt(self.test_variance[None, :, None] + self.epsilon)) * self.gamma[None, :, None] + self.beta[None, :, None]

    def Forward_training(self, inp):
        N, _, W = inp.shape
        self.NW = N * W #Couldn't come up with a better name for it

        mean = np.sum(inp, axis=(0, 2)) / self.NW 
 
        variance = np.sum(np.square(inp-mean), axis=(0, 2)) / self.NW

        self.cache0 = inp - mean[None, :, None]
        self.cache1 = 1 / np.sqrt(variance + self.epsilon)
        self.cache2 = self.cache0 * self.cache1[None, :, None] * self.gamma[None, :, None]

        self.test_mean = (self.test_mean * self.decay_rate) + ((1-self.decay_rate) * mean)
        self.test_variance = (self.test_variance * self.decay_rate) + ((1-self.decay_rate) * variance)
    
        return self.cache2 + self.beta[None, :, None]

    def Backward(self, grad):
        variance_grad = np.sum(self.cache0 * (-(self.gamma * np.cube(self.cache1, 3))/2)[None, :, None], axis=(0, 2))
        mean_grad = np.sum((((self.gamma * self.cache1)/2)[None, :, None] * -grad), axis=(0, 2)) + ((-2 * variance_grad * np.sum(self.cache0, axis=(0, 2)))/self.NW)
        
        #Update beta and gamma
        self.beta -= self.optimizer.use(np.sum(grad, axis=(0, 2)) / self.NW)
        self.gamma -= self.optimizer.use(np.sum(grad * self.cache2, axis=(0, 2)) / self.NW)
        
        return (grad * self.cache1[None, :, None]) + ((variance_grad[None, :, None] * 2 * self.cache0)/self.NW) + (mean_grad/self.NW)[None, :, None]

    def Build(self):
        pass

    def Save(self):
        return {'args':(), 'var':(self.decay_rate, self.epsilon, self.gamma, self.beta, self.test_mean, self.test_variance, self.optimizer)}

    def Load(self, var):
        self.decay_rate, self.epsilon, self.gamma, self.beta, self.test_mean, self.test_variance, self.optimizer = var