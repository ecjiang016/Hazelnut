import numpy as np

def NoOptimizer(learning_rate, weight_gradient, cache=None, hyperparameter=None):
    return weight_gradient * learning_rate, None

def Momentum(learning_rate, weight_gradient, cache=0, hyperparameter=0.9): #Hyperparameter (beta) set at 0.9 or 0.999 for extremely noisy loss funcntion
    momentum_vector = (cache * hyperparameter) + (1 - hyperparameter) * weight_gradient
    return momentum_vector * learning_rate, momentum_vector

def RProp(learning_rate, weight_gradient, cache=(1, 0), hyperparameter=((0.5, 1.2), (0.000001, 50))):
    """
    Using RPROP (Resilient propagation) for backpropagation
    Note: Cache some step size > 0 and 0 in a tuple in the optimizer beforehand as the starting step size and signs. Ex: (25, 0)

    cache: Previous step sizes (ndarray)
    hyperparameter: Min and max step sizes. Step sizes get clipped to be within this range
    """

    scale, clip = hyperparameter
    scale_small, scale_large = scale
    clip_min, clip_max = clip
    previous_steps, previous_signs = cache

    signs = np.sign(weight_gradient)
    out = previous_steps * signs

    sign_change = previous_signs * signs
    new_steps = np.clip((((sign_change < 0) * scale_small) + ((sign_change > 0) * scale_large) + ((sign_change == 0) * 1)) * previous_steps, clip_min, clip_max)
    return out, (new_steps, signs)
