from .Reformatting import col2im, col2im_back
from .Reformatting import memory_strided_im2col as im2col
import numpy as np

def conv(inp, kern):
    """
    Convolution by conversion into matrix multipication through im2col

    Input:
    - inp: Input data of shape (N, C, H, W)
    - kern: Kernels/filters in the shape (F, C, HH, WW)

    Returns the input convolved by the kernel
    """
    N, C, H, W = inp.shape
    F, C, HH, WW = kern.shape
    H_prime = H - HH + 1
    W_prime = W - WW + 1

    inp_cols = np.array([im2col(tensor, HH) for tensor in inp])
    inp_col = np.hstack(inp_cols)

    kern_col = np.reshape(np.flip(kern, (2, 3)), (F, -1))

    mul = np.matmul(kern_col, inp_col)

    single_output_len = H_prime * W_prime

    output = []
    for i in range(N):
        index = i*single_output_len
        output.append(col2im(mul[:, index:index+single_output_len], H_prime, W_prime, C, F))

    return np.array(output)
        
def conv_kern_grad(acti, grad):
    """
    Custom function for calculating kernel gradients.
    Optimized by summing the gradients during the matrix multiplication and sacrificing memory.

    Input:
    - acti: The cached activations of shape (N, C, H, W)
    - grad: The previous gradient of shape (N, C, H, W)

    Returns the kernel gradients
    """

    F, C, HH, WW = acti.shape
    N, C, H, W = grad.shape

    for c in range(C):
        inp_cols = np.array([im2col(grad[n, c], HH) for n in range(N)])
        inp_col = np.hstack(inp_cols)

def conv_full(inp, kern):
    """
    Full convolution through im2col.

    - inp: Input data of shape (N, C, H, W)
    - kern: Kernels/filters in the shape (F, C, HH, WW)

    Returns the input convolved by the kernel
    """

    N, C, H, W = inp.shape
    F, C, HH, WW = kern.shape
    H_prime = H + HH - 1
    W_prime = W + WW - 1

    #padded_inp = np.zeros((N, C, H+HH-1, W+WW-1))
    #for n in range(N): #Sucky nested loop but idk what else to do
        #for c in range(C):
            #padded_inp[n, c] = np.pad(inp[n, c], HH-1)

    padded_inp = np.array([[np.pad(inp[n, c], HH-1) for c in range(C)] for n in range(N)])
    
    inp_cols = np.array([im2col(tensor, HH) for tensor in padded_inp])
    inp_col = np.hstack(inp_cols)

    kern_col = np.reshape(np.flip(kern, (2, 3)), (F, -1))

    mul = np.matmul(kern_col, inp_col)

    single_output_len = H_prime * W_prime

    output = []
    for i in range(N):
        index = i*single_output_len
        output.append(col2im(mul[:, index:index+single_output_len], H_prime, W_prime, C, F))

    return np.array(output)

def conv_backward_naive(x, w, mode):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    C,H,W = x.shape
    F,C,HH,WW = w.shape

    if mode == 'valid':
        pad_num = 0
    elif mode == 'same':
        pad_num = (HH-1) // 2
    elif mode == 'full':
        pad_num = HH-1
    else:
        raise ValueError("Not a valid mode")

    pad_num = 0
    stride = 1
    H_prime = (H+2*pad_num-HH) // stride + 1
    W_prime = (W+2*pad_num-WW) // stride + 1

    #im2col
    im_pad = np.pad(x, ((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
    im_col = im2col(im_pad,HH,WW,stride)
    filter_col = np.reshape(w,(F,-1))
    mul = im_col.dot(filter_col.T)
    return col2im(mul,H_prime,W_prime,1)

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    x, w, b, conv_param = cache
    pad_num = conv_param['pad']
    stride = conv_param['stride']
    N,C,H,W = x.shape
    F,C,HH,WW = w.shape
    H_prime = (H+2*pad_num-HH) // stride + 1
    W_prime = (W+2*pad_num-WW) // stride + 1

    dw = np.zeros(w.shape)
    dx = np.zeros(x.shape)
    db = np.zeros(b.shape)

    # We could calculate the bias by just summing over the right dimensions
    # Bias gradient (Sum on dout dimensions (batch, rows, cols)
    #db = np.sum(dout, axis=(0, 2, 3))

    for i in range(N):
        im = x[i,:,:,:]
        im_pad = np.pad(im,((0,0),(pad_num,pad_num),(pad_num,pad_num)),'constant')
        im_col = im2col(im_pad,HH,WW,stride)
        filter_col = np.reshape(w,(F,-1)).T

        dout_i = dout[i,:,:,:]
        dbias_sum = np.reshape(dout_i,(F,-1))
        dbias_sum = dbias_sum.T

        #bias_sum = mul + b
        db += np.sum(dbias_sum,axis=0)
        dmul = dbias_sum

        #mul = im_col * filter_col
        dfilter_col = (im_col.T).dot(dmul)
        dim_col = dmul.dot(filter_col.T)

        dx_padded = col2im_back(dim_col,H_prime,W_prime,stride,HH,WW,C)
        dx[i,:,:,:] = dx_padded[:,pad_num:H+pad_num,pad_num:W+pad_num]
        dw += np.reshape(dfilter_col.T,(F,C,HH,WW))
    return dx, dw, db