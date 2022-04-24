class SGD:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    
    def use(self, gradient):
        return gradient * self.learning_rate

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class SGDM:
    """
    Stochastic Gradient Descent + Momentum
    Momentum in the form:
    learning_rate * gradient - (momentum_weight * (gradient - previous_gradient))
    """
    def __init__(self, learning_rate=1e-5, momentum_weight=1e-5) -> None:
        self.learning_rate = learning_rate
        self.momentum_weight = momentum_weight

        self.update_cache = 0

    def use(self, gradient):
        out = self.learning_rate * gradient + (self.momentum_weight * self.update_cache)
        self.update_cache = out.copy()
        return out
        
    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.momentum_weight, self.gradient_cache)}

    def Load(self, var):
        self.learning_rate, self.momentum_weight, self.gradient_cache = var

class Momentum:
    def __init__(self, learning_rate=0.0001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta

        self.momentum_vector = 0

    def use(self, gradient):
        self.momentum_vector = (self.momentum_vector * self.beta) + (1 - self.beta) * gradient
        return self.momentum_vector * self.learning_rate

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

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

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        new_signs = self.np.sign(gradient)
        sign_change = self.signs * new_signs
        self.steps = self.np.clip((((sign_change < 0) * self.scale_small) + ((sign_change > 0) * self.scale_large) + ((sign_change == 0) * 1)) * self.steps, self.clip_min, self.clip_max)
        self.signs = new_signs
        return self.steps * new_signs

    def Save(self):
        return {'args':(), 'var':(self.steps, self.scale_large, self.scale_large, self.clip_min, self.clip_max, self.signs)}

    def Load(self, var):
        self.steps, self.scale_large, self.scale_large, self.clip_min, self.clip_max, self.signs = var

class RMSProp:
    """Root mean squared propagation"""
    def __init__(self, learning_rate=1e-6, beta=0.9) -> None:
        """
        Args:
        - learning_rate: Learning rate
        - beta: The weight of the accumulated squared gradients over the new one

        Hidden:
        - epsilon: Small number stopping division by zero. Automatically set to 1e-10
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-10

        #Initial values
        self.grad_average = 0

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        self.grad_average = self.beta * self.grad_average + ((1 - self.beta) * self.np.square(gradient))
        
        return gradient * self.learning_rate / self.np.sqrt(self.grad_average + self.epsilon)

    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.beta, self.epsilon, self.grad_average)}

    def Load(self, var):
        self.learning_rate, self.beta, self.epsilon, self.grad_average = var

class Adam:
    """Adaptive Moment Optimization"""
    def __init__(self, learning_rate=1e-6, beta1=0.9, beta2=0.99) -> None:
        """
        Args:
        - learning_rate: Learning rate
        - beta1: The weight of the accumulated gradients over the new one
        - beta2: The weight of the accumulated squared gradients over the new one

        Hidden:
        - epsilon: Small number stopping division by zero. Automatically set to 1e-10
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-10

        #Initial values
        self.grad_average = 0
        self.square_grad_average = 0

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        self.grad_average = self.beta1 * self.grad_average + ((1 - self.beta1) * gradient)
        self.square_grad_average = self.beta2 * self.square_grad_average + ((1 - self.beta2) * self.np.square(gradient))
        
        return gradient * self.grad_average * self.learning_rate / self.np.sqrt(self.square_grad_average + self.epsilon)

    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.grad_average, self.square_grad_average)}

    def Load(self, var):
        self.learning_rate, self.beta1, self.beta2, self.epsilon, self.grad_average, self.square_grad_average = var
