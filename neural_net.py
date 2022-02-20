import pickle
import numpy as np
from .modules.Reshape import Convert4Dto3D, Convert3Dto4D

class NN:
    def __init__(self):
        pass

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
        self.layout.append(module)

    def optimizer(self, optimizer) -> None:
        for module in self.layout:
            try:
                module.optimizer
                module.optimizer = optimizer
            except AttributeError:
                pass

    def build(self, inp_size) -> None:
        """
        Builds each layer given the constant input shape to the net

        Args:
        - inp_size: A tuple with elements (channels, height, width) or (size). The first for CNN and the latter for MLP
        """
        self.layout = []

        end_reshape = True

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

            if pass_inp.shape != inp_size: #If there is some kind of reshape, don't reshape the activations at the end
                end_reshape = False

        if end_reshape: #Reshape back from (N, C, H*W) to (N, C, H, W)
            self.layout.append(Convert3Dto4D(H, W))


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
