import numpy as np

class MSE:
    """
    Mean squared error
    """
    def Forward(out, correct_out):
        return np.sum(np.square(out-correct_out)) / out.shape[0]

    def Backward(out, correct_out):
        return out - correct_out

class CrossEntropy:
    """
    Note: Also does a softmax before applying the crosss entropy loss
    """
    def Forward(out, correct_out):
        #Softmax time
        exps = np.exp(out - np.max(out, axis=0)) #Add numerical stability
        softmax_out = exps/np.sum(exps, axis=0)
        return np.sum(correct_out * np.log(softmax_out)) / out.shape[0]

    def Backward(out, correct_out):
        return out - correct_out