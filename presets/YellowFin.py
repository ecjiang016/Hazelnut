from ..neural_net import NN as Base_NN
from ..modules.Optimizers import SGDM

class NN(Base_NN):
    """
    Neural net with YellowFin
    """
    def __init__(self, beta=0.999, slow_start=True, gradient_clip=True, sliding_window=20):
        """
        Make sure to super().__init__()

        Variables:
        - self.beta: Default value 0.9
        - self.epsilon: For numerical stability. Default value 1e-7
        """
        super().__init__()
        self.optimizer = SGDM()

        #YellowFin variables
        self.beta = beta
        self.epsilon = 1e-7
        self.slow_start = slow_start
        self.gradient_clip = gradient_clip
        self.sliding_window = sliding_window

        #Iteration tracking
        self.t = 0

        #Curvature range variables
        self.h = []
        self.h_max = 0
        self.h_min = 0

        #Gradient variance variables
        self.g_2 = 0
        self.g_ = 0

        #Distance to optimum variables
        self.g_norm = 0
        self.h_ = 0
        self.D = 0

    def update(self):
        grad = []
        for module in self.layout:
            try:
                module_gradient = module.gradient
                if type(module_gradient) is tuple:
                    for gradient in module_gradient:
                        grad.append(self.np.ravel(gradient))

                else:
                    grad.append(self.np.ravel(module_gradient))
            except AttributeError:
                pass
    
        grad = self.np.concatenate(grad)
        grad_norm2 = float(self.np.square(grad).sum())

        #Calculate curvature range
        self.h.append(grad_norm2)
        if len(self.h) > self.sliding_window:
            del self.h[0]

        h_max_t = max(self.h)
        h_min_t = min(self.h)

        if self.gradient_clip:
            self.h_max = (self.beta * self.h_max) + ((1 - self.beta) * min(h_max_t, 100 * self.h_max))
        else:
            self.h_max = (self.beta * self.h_max) + ((1 - self.beta) * h_max_t)
        self.h_min = (self.beta * self.h_min) + ((1 - self.beta) * h_min_t)

        #Gradient variance
        self.g_2 = (self.beta * self.g_2) + ((1 - self.beta) * self.np.square(grad))
        self.g_ = (self.beta * self.g_) + ((1 - self.beta) * grad)
        C = float(self.np.sum(self.g_2 - self.np.square(self.g_)))

        #Distance to optimum
        self.g_norm = (self.beta * self.g_norm) + ((1 - self.beta) * self.np.sqrt(grad_norm2))
        self.h_ = (self.beta * self.h_) + ((1 - self.beta) * grad_norm2)
        self.D = (self.beta * self.D) + ((1 - self.beta) * (self.g_norm/self.h_))

        #SingleStep
        
        #Get cubic root
        #From https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py
        p = (self.D + self.epsilon)**2 * (self.h_min + self.epsilon)**2 / 2 / (C + self.epsilon)
        w3 = (-self.np.sqrt(p**2 + 4.0 / 27.0 * p**3) - p) / 2.0
        w = self.np.sign(w3) * self.np.power(self.np.abs(w3), 1.0/3.0)
        y = w - p / 3.0 / (w + self.epsilon)
        x = y + 1

        root = x ** 2
        dr = self.np.maximum( (self.h_max + self.epsilon) / (self.h_min + self.epsilon), 1.0 + self.epsilon)
        momentum = self.np.maximum(root, ((self.np.sqrt(dr) - 1) / (self.np.sqrt(dr) + 1))**2)
        learning_rate = self.np.square(1 - self.np.sqrt(momentum)) / self.h_min

        if self.slow_start:
            learning_rate = min(learning_rate, self.t * learning_rate / (10 * self.sliding_window))

        #Update all the optimizers
        for module in self.layout:
            try:
                module.optimizer.learning_rate = learning_rate
                module.optimizer.momentum_weight = momentum
            except AttributeError: #Module doesn't have an optimizer
                pass

        self.t += 1
        
        super().update()