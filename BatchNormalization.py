import numpy as np
from .Optimizers import *

def FeedForward(inp, parameters, training=False):
    """
    Args:
    - inp: input array (4D tensor)
    - parameters: Learned parameters Gamma and Beta where (gamma * normalized_input) + beta and the exponential running average for the mean and variance.

    Returns a tuple including:
    - A numpy array with the same shape as the input array 
    - Cache with a tuple containing inp - mean, 1 / sqrt(variance + epsilon), and gamma * (inp-mean)/(sqrt(variance + epsilon))
    """

    N, C, H, W = inp.shape
    gamma, beta, test_mean, test_variance = parameters
    epsilon = 0.01 #Constant that helps with stability by preventing division by zero

    if training:
        mean = np.zeros((N, C))
        variance = np.zeros((N, C))
        channel_size = inp[0, 0].size

        for n in range(N):
            for c in range(C):
                mean[n, c] = np.sum(inp[n, c]) / channel_size
                variance[n, c] = np.square(inp - mean[n, c]) / channel_size
    
    else:
        mean = test_mean
        variance = test_variance

    output = np.zeros((N, C, H, W))
    cache0 = output.copy()
    cache1 = output.copy()
    cache2 = output.copy()
    for n in range(N):
        for c in range(C):
            cache0[n, c] = inp[n, c] - mean[n, c]
            cache1[n, c] =  1 / np.sqrt(variance[n, c] + epsilon)
            cache2[n, c] = cache0[n, c] * cache1[n, c] * gamma[c]
            output[n, c] = cache2[n, c] + beta[c]
    
    if training: #Update the running mean of the mean and variance
        decay_rate = 0.9 #Constant determining how important the latest mean and variance is in the mean
        test_mean = (test_mean * decay_rate) + ((1-decay_rate) * mean)
        test_variance = (test_variance * decay_rate) + ((1-decay_rate) * variance)
    
    return output, (cache0, cache1, cache2)

def Backpropagate(grad, parameters, cache, learning_rate): #Need to implement optimizers
    """
    Args:
    - grad: previous gradient (4D tensor)
    - parameters: Learned parameters Gamma and Beta where (gamma * normalized_input) + beta and the exponential running average for the mean and variance.
     - Cache with a tuple containing inp - mean, 1 / sqrt(variance + epsilon), and gamma * (inp-mean)/(sqrt(variance + epsilon))

    Returns:
     - The gradient to pass on as numpy array with the same shape as the input array 
    """
    cache0, cache1, cache2 = cache

    N, C, H, W = grad.shape
    gamma, beta, _, _ = parameters

    beta_grad = sum(grad)
    gamma_grad = sum(grad * cache2)
    for c in range(C): #Update beta and gamma
        beta[c] -= np.sum(beta_grad[c]) * learning_rate
        gamma[c] -= np.sum(gamma_grad[c]) * learning_rate

    pass_gradient = np.zeros((N, C, H, W))
    for n in range(N):
        for c in range(C):
            variance_grad = np.sum(cache0[n, c] * (-(gamma[c] * (cache1[n, c]**3))/2))
            mean_grad = np.sum(((-(gamma[c] * cache1[n, c])/2) * grad[n, c])) + ((-2 * variance_grad[c] * np.sum(cache0[n, c]))/C)
            pass_gradient[n, c] = (grad[n, c] * cache1[n, c]) + ((variance_grad * 2 * cache0[n, c])/C) + (mean_grad/C)
    
    return pass_gradient
