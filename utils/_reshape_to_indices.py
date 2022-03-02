import numpy as np

def reshape_to_indices(function):
    """
    Takes a function that reshapes an array
    and returns indices that will also result in that reshape

    Usage: Pass the function with any array with the desired dimensions.
    """

    def inner(arg):
        arange = np.arange(arg.size).reshape(arg.shape)
        out = function(arange)

        indices = []
        div = arg.size
        for mod in arg.shape:
            div = div // mod
            indices.append(out // div % mod)

        return indices

    return inner

if __name__ == '__main__':
    @reshape_to_indices
    def r(arr):
        return np.flip(arr, 1)
    
    indices = r(np.zeros((3, 2, 4, 4)))

    inp = np.random.randint(-5, 6, size=(3, 2, 4, 4))
    print(np.flip(inp, 1) == inp[indices])
