import numpy as np

def He_init(F, C, KS):
    return np.random.normal(0, np.sqrt(2/(F * C * KS * KS)), size=(F, C, KS, KS))

def Xavior_init(F, C, KS):
    return np.random.normal(0, np.sqrt(1/(F * C * KS * KS)), size=(F, C, KS, KS))