import numpy as np

def FeedForward(inp, parameters, training=False):
    """
    Args:
    - inp: input array (4D tensor)
    - parameters: Learned parameters Gamma and Beta where (gamma * normalized_input) + beta and the exponential running average for the mean and variance.

    Returnsa tuple including:
    - A numpy array with the same shape as the input array 
    """

    N, C, H, W = inp.shape
    gamma, beta, mean, variance = parameters
    epsilon = 0.1 #Constant that helps with stability by preventing division by zero

    if training:
        mean = np.zeros((N, C))
        variance = np.zeros((N, C))
        channel_size = inp[0, 0].size

        for n in range(N):
            for c in range(C):
                mean[n, c] = np.sum(inp[n, c]) / channel_size
                variance[n, c] = np.square(inp - mean[n, c]) / channel_size

    output = np.zeros((N, C, H, W))
    for n in range(N):
        for c in range(C):
            output[n, c] = (inp[n, c] - mean[n, c]) / np.sqrt(variance[n, c] + epsilon) * gamma + beta
    
    return output