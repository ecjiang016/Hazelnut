import pickle
import numpy as np
from .modules.reshape import Convert4Dto3D

class NN:
    def __init__(self):
        self.layout = []

    def __repr__(self) -> str:
        str_ = ""
        for module in self.layout:
            str_ += str(module.__class__.__name__) + "\n"

        return str_[:-1]

    def __str__(self) -> str:
        str_ = ""
        for module in self.layout:
            str_ += str(module.__class__.__name__) + "\n"

        return str_[:-1]

    def forward(self, inp):
        for module in self.layout:
            inp = module.Forward(inp)

        return inp

    def forward_train(self, inp):
        for module in self.layout:
            inp = module.Forward_training(inp)

        return inp

    def backpropagate(self, grad):
        for module in reversed(self.layout):
            grad = module.Backward(grad)

    def train(self, inp, correct_out):
        out = self.forward_train(inp)
        self.backpropagate(self.loss.Backward(out, correct_out))

        return self.loss.Forward(out, correct_out)

    def add(self, module):
        self.layout.append(module)

    def optimizer(self, optimizer):
        for module in self.layout:
            try:
                module.optimizer
                module.optimizer = optimizer
            except AttributeError:
                pass

    def build(self, inp_size):
        """
        Builds each layer given the constant input shape to the net

        Args:
        - inp_size: A tuple with elements (channels, height, width)
        """
        if len(inp_size) == 4:
            #If CNN, the inputs need to be reshaped to (1, C, H*W) as that's how the modules takes care of them
            C, H, W = inp_size.shape
            pass_inp = np.zeros((1, C, H * W))

            self.layout.insert(0, Convert4Dto3D())

        elif len(inp_size) == 1:
            #MLP ig
            H = inp_size
            pass_inp = np.zeros((H, 1))

        else:
            raise ValueError("I have no idea what you're doing with the input size")

        for module in self.layout:
            module.Build(pass_inp)
            pass_inp = module.Forward(pass_inp)

    def save(self, PATH):
        save_list = []
        for module in self.layout:
            save_dict = module.Save()
            save_list.append({'module_class':module.__class__, 'save_dict':save_dict})

        f = open(PATH, "wb")
        pickle.dump(save_list, f)
        f.close()

    def load(self, PATH):
        f = open(PATH, "rb")
        save_list = pickle.load(f)
        f.close()

        self.layout = []
        for save_module_dict in save_list:
            module_class = save_module_dict['module_class']
            save_dict = save_module_dict['save_dict']

            args = save_dict['args']
            var = save_dict['var']

            module = module_class(*args)
            module.load(var)
            self.layout.append(module)
