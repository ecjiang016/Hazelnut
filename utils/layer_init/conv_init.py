import numpy as np

#Note: Keep these as functions that return numpy arrays and convert to cupy in the modules if needed

def He_init(F, C, KS) -> np.ndarray:
    return np.random.normal(0, np.sqrt(2/(F * C * KS * KS)), size=(F, C, KS, KS))

def Xavior_init(F, C, KS) -> np.ndarray:
    return np.random.normal(0, np.sqrt(1/(F * C * KS * KS)), size=(F, C, KS, KS))