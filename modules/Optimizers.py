import numpy as np

class NoOptimizer:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    
    def use(self, gradient):
        return gradient * self.learning_rate

class Momentum:
    def __init__(self, learning_rate=0.0001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta

        self.momentum_vector = 0

    def use(self, gradient):
        self.momentum_vector = (self.momentum_vector * self.beta) + (1 - self.beta) * gradient
        return self.momentum_vector * self.learning_rate
class RProp:
    def __init__(self, step_size=0.001, scale=(0.5, 1.2), clip=(0.000001, 50)):
        """
        - step_size: starting step size > 0
        - scale: The scaling sizes for the step sizes. A tuple with (decrease scale, increase scale)
        - clip: The min and max step size. A tuple with (min size, max size)
        """
        self.steps = step_size
        self.scale_small, self.scale_large = scale
        self.clip_min, self.clip_max =  clip
        self.signs = 0

    def use(self, gradient):
        new_signs = np.sign(gradient)
        sign_change = self.signs * new_signs
        self.steps = np.clip((((sign_change < 0) * self.scale_small) + ((sign_change > 0) * self.scale_large) + ((sign_change == 0) * 1)) * self.steps, self.clip_min, self.clip_max)
        self.signs = new_signs
        return self.steps * new_signs