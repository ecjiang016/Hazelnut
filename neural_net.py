import pickle
import numpy as np
from copy import deepcopy

class NN:
    def __init__(self):
        self.layout = []
        self.loss = None
        self.optimizer = None

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

    def add(self, module) -> None:
        try:
            self.layout.append(module)
        except AttributeError: #If self.layout doesn't exist, make one
            self.layout = []
            self.layout.append(module)


    def build(self, inp_size) -> None:
        """
        Builds each layer given the constant input shape to the net

        Args:
        - inp_size: A tuple with elements (channels, height, width) or (size). The first for CNN and the latter for MLP
        """
        assert self.loss, "No loss function specified"
        assert self.optimizer, "No optimizer specified"

        if len(inp_size) == 3:
            #If CNN, the inputs need to be reshaped to (1, C, H*W) as that's how the modules takes care of them
            C, H, W = inp_size
            pass_inp = np.zeros((1, C, H, W))

        elif len(inp_size) == 1:
            #MLP ig
            H = inp_size
            pass_inp = np.zeros((H, 1))

        else:
            raise ValueError("I have no idea what you're doing with the input size")

        for module in self.layout:
            try:
                module.optimizer
                module.optimizer = deepcopy(self.optimizer)
            except AttributeError: #Module doesn't need an optimizer
                pass
        
            module.Build(pass_inp.shape)
            pass_inp = module.Forward(pass_inp)
            if np.isnan(np.min(pass_inp)):
                raise SystemError("NaN incountered.")


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
