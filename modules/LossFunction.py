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
    Note: Needs a Softmax layer before to work properly
    """
    def __init__(self) -> None:
        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, out, correct_out):
        return -self.np.sum(correct_out * self.np.log(out)) / out.shape[1]

    def Backward(self, out, correct_out):
        return out - correct_out