import numpy as np

function_names = ['AF', 'Sigmoid', 'tanh', 'ReLU', 'LeakyReLU', 'step']
function_names = function_names + [name+"_dv" for name in function_names]
__all__ = function_names

def AF(Vector, Type):
    return globals()[Type](Vector)

def AF_dv(Vector, Type):
    return globals()[Type+"_dv"](Vector)

def Sigmoid(x):
    return 1/(np.exp(-x)+1)

def Sigmoid_dv(x):
    return Sigmoid(x)*(1-Sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_dv(x):
    return 1-(np.tanh(x)**2)

def ReLU(x):
    if x > 0:
         return x
    else:
        return 0

def ReLU_dv(x):
    if x >= 0:
        return 1
    return 0

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

def Softmax_single(x):
    numerator = np.exp(x)
    denominator = np.sum(numerator)
    return np.array([numerator[element]/denominator for element in range(len(x))])

def Softmax(x):
    vectors = x.T
    return np.array([Softmax_single(vectors[i]) for i in range(len(vectors))]).T

def Softmax_single_dv(x, y):
    return np.matmul((np.diag(y.T[0]) - np.matmul(y, y.T)), x).T[0]

#def Softmax_dv(x):
    #vectors = x.T
    #const = Activations[LastLayer].T
    #return np.array([Softmax_dv_single(np.array([vectors[i]]).T, np.array([const[i]]).T) for i in range(len(vectors))]).T

def step(x):
    if x >= 0.5:
        return 1
    elif x < 0.5:
        return 0

def step_dv(x):
    return 0

ReLU = np.vectorize(ReLU, otypes=[np.float64])
ReLU_dv = np.vectorize(ReLU_dv, otypes=[np.float64])
LeakyReLU = np.vectorize(LeakyReLU, otypes=[np.float64])
LeakyReLU_dv = np.vectorize(LeakyReLU_dv, otypes=[np.float64])
step = np.vectorize(step)