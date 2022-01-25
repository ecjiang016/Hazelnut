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

    gamma_4D = np.zeros((N, C, H, W))
    beta_4D = np.zeros((N, C, H, W))

    for c in range(C):
        gamma_4D[:, c] = gamma[c]
        beta_4D[:, c] = beta[c]

    if training:
        channel_size = H * W
        
        mean = np.sum(inp, axis=(2, 3)) / channel_size
        mean = np.swapaxes(np.swapaxes(np.full((H, W, N, C), mean), 0, 2), 1, 3)

        variance = np.sum(np.square(inp-mean), axis=(2, 3)) / channel_size
        variance = np.swapaxes(np.swapaxes(np.full((H, W, N, C), variance), 0, 2), 1, 3)
    
    else:
        mean = test_mean
        variance = test_variance

    cache0 = inp - mean
    cache1 = 1 / np.sqrt(variance + epsilon)
    cache2 = cache0 * cache1 * gamma_4D
    output = cache2 + beta_4D
    
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
    gamma, beta, test_mean, test_variance = parameters

    gamma_4D = np.zeros((N, C, H, W))
    beta_4D = np.zeros((N, C, H, W))

    for c in range(C):
        gamma_4D[:, c] = gamma[c]
        beta_4D[:, c] = beta[c]

    #Update beta and gamma
    beta -= np.sum(grad, axis=(0, 2, 3)) * learning_rate
    gamma -= np.sum(grad * cache2, axis=(0, 2, 3)) * learning_rate

    channel_size = H * W

    variance_grad = np.sum(cache0 * (-(gamma_4D * (cache1**3))/2), axis=(2, 3))
    mean_grad = np.sum(((-(gamma_4D * cache1)/2) * grad), axis=(2, 3)) + ((-2 * variance_grad * np.sum(cache0, axis=(2, 3)))/channel_size)

    variance_grad = np.swapaxes(np.swapaxes(np.full((H, W, N, C), variance_grad), 0, 2), 1, 3)
    mean_grad = np.swapaxes(np.swapaxes(np.full((H, W, N, C), mean_grad), 0, 2), 1, 3)

    pass_gradient = (grad * cache1) + ((variance_grad * 2 * cache0)/channel_size) + (mean_grad/channel_size)
    
    return pass_gradient, (gamma, beta, test_mean, test_variance)
