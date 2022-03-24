from ..neural_net import NN as Base_NN
from ..modules.Optimizers import SGDM

class NN(Base_NN):
    """
    Neural net with YellowFin
    """
    def __init__(self):
        """
        Make sure to super().__init__()

        Variables:
        - self.beta: Default value 0.9
        - self.epsilon: For numerical stability. Default value 1e-7
        """
        super().__init__()
        self.optimizer = SGDM

        #YellowFin variables
        self.beta = 0.9
        self.epsilon = 1e-7

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
    
    def train(self, inp, correct_out):
        out = self.forward_train(inp)
        back_gradient = self.loss.Backward(out, correct_out)
        gradient = self.np.sum(back_gradient, axis=1)[:, None] / back_gradient.shape[1]

        #Calculate learning rate and momentum weight

        #Calculate curvature range
        self.h.append(float(self.np.sum(self.np.square(gradient))))
        if len(self.h) > 50:
            self.h.pop(0)

        h_max_t = max(self.h)
        h_min_t = min(self.h)

        self.h_max = (self.beta * self.h_max) + ((1 - self.beta) * h_max_t)
        self.h_min = (self.beta * self.h_min) + ((1 - self.beta) * h_min_t)

        #Gradient variance
        self.g_2 = (self.beta * self.g_2) + ((1 - self.beta) * self.np.square(gradient))
        self.g_ = (self.beta * self.g_) + ((1 - self.beta) * gradient)
        C = float(self.np.sum(self.g_2 - self.np.square(self.g_)))

        #Distance to optimum
        self.g_norm = (self.beta * self.g_norm) + ((1 - self.beta) * float(self.np.sum(self.np.absolute(self.g_norm))))
        self.h_ = (self.beta * self.h_) + ((1 - self.beta) * self.h[-1])
        self.D = (self.beta * self.D) + ((1 - self.beta) * (self.g_norm/self.h_))


        #SingleStep
        
        #Get cubic root
        #From https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py
        p = (self.D + self.epsilon)**2 * (self._h_min + self.epsilon)**2 / 2 / (self._grad_var + self.epsilon)
        w3 = (-self.np.sqrt(p**2 + 4.0 / 27.0 * p**3) - p) / 2.0
        w = self.np.sign(w3) * self.np.pow(self.np.abs(w3), 1.0/3.0)
        y = w - p / 3.0 / (w + self.epsilon)
        x = y + 1

        root = self.get_cubic_root() ** 2
        dr = self.np.maximum( (self._h_max + self.epsilon) / (self._h_min + self.epsilon), 1.0 + self.epsilon)
        learning_rate = self.np.maximum(root, ((self.np.sqrt(dr) - 1) / (self.np.sqrt(dr) + 1))**2)
        momentum = self.np.square(1 - self.np.sqrt(learning_rate)) / self.h_min

        #Update all the optimizers
        for module in self.layout:
            try:
                module.optimizer.learning_rate = learning_rate
                module.optimizer.momentum_weight = momentum
            except AttributeError: #Module doesn't have an optimizer
                pass

        self.backpropagate(back_gradient)
        loss = self.loss.Forward(out, correct_out)

        if self.np.isnan(loss):
            raise RuntimeError("Loss is NaN")

        return loss, out