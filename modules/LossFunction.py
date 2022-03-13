import numpy

class MSE:
    """
    Mean squared error
    """
    def __init__(self) -> None:
        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, out, correct_out):
        return self.np.sum(self.np.square(out-correct_out)) / out.shape[1]

    def Backward(self, out, correct_out):
        return out - correct_out

class CrossEntropy:
    """
    Note: Also does a softmax before applying the crosss entropy loss
    """
    def __init__(self) -> None:
        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, out, correct_out):
        #Softmax time
        exps = self.np.exp(out - self.np.max(out, axis=0)) #Add numerical stability
        softmax_out = exps/self.np.sum(exps, axis=0)
        return -self.np.sum(correct_out * self.np.log(softmax_out)) / out.shape[1]

    def Backward(self, out, correct_out):
        return out - correct_out