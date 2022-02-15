from re import X
import numpy as np

function_names = ['AF', 'Sigmoid', 'Softmax', 'tanh', 'ReLU', 'LeakyReLU', 'step']
function_names = function_names + [name+"_dv" for name in function_names]
__all__ = function_names

def AF(Vector, Type):
    return globals()[Type](Vector)

def AF_dv(Vector, Type):
    return globals()[Type+"_dv"](Vector)

def Sigmoid(x):
    clipped_x = np.clip(x, -200, 200)
    return 1/(np.exp(-clipped_x)+1)

def Sigmoid_dv(x):
    return Sigmoid(x)*(1-Sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_dv(x):
    return 1-(np.tanh(x)**2)

def ReLU(x):
    out = x.copy()
    out[out < 0] = 0
    return out

def ReLU_dv(x):
    return (x >= 0) * 1

def LeakyReLU(x):
    if x > 0:
        return x
    elif x <= 0:
        return 0.2*x
    
def LeakyReLU_dv(x):
    if x > 0:
        return 1
    elif x <= 0:
        return 0.2

def Softmax(x):
    exps = x - np.max(x, axis=0) #Add numerical stability
    return exps/np.sum(exps, axis=0)

def Softmax_dv(x):
    pass

def step(x):
    if x >= 0.5:
        return 1
    elif x < 0.5:
        return 0

def step_dv(x):
    return 0

def identity(x):
    return x

def identity_dv(x):
    return 1

LeakyReLU = np.vectorize(LeakyReLU, otypes=[np.float64])
LeakyReLU_dv = np.vectorize(LeakyReLU_dv, otypes=[np.float64])
step = np.vectorize(step)