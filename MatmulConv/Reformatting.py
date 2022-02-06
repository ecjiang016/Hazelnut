import numpy as np
from numpy.lib.stride_tricks import as_strided
#from skimage.util.shape import view_as_windows

def memory_strided_im2col(x, KS): 
    """
    Args:
    x: image matrix to be translated into columns, (N,C,H,W)
    KS: length of a side of a square kernel

    Returns:
    A (C*KS*KS, N*OH*OW) matrix
    """
    N, C, H, W = x.shape
    OH = H - KS + 1 #Output height
    OW = W - KS + 1 #Output width
    SN, SC, SH, SW = x.strides
    return as_strided(x,
               shape=(C, KS, KS, N, OH, OW),
               strides=(SC, SH, SW, SN, SH, SW),
               ).reshape((C*KS*KS, N*OH*OW))

def skimage_strided_im2col(x, kernel_size):
    """
    Args:
    x: image matrix to be translated into columns, (C,H,W)
    kernel_size: length of a side of a square kernel

    Returns:
    col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
    new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    #output_shape = x.shape[1] - kernel_size + 1
    #cols = view_as_windows(x, (1, kernel_size, kernel_size))
    #output = [cols[i].reshape(output_shape*output_shape, kernel_size*kernel_size).T for i in range(x.shape[0])]
    #return np.vstack(output)

def memory_strided_im2col_single(x, kernel_size):
    """
    Same thing as memory_strided_im2col but for images with no channels for slightly better preformance

    Args:
    x: image matrix to be translated into columns, (H,W)
    kernel_size: length of a side of a square kernel

    Returns:
    col: (new_h*new_w,hh*ww) matrix, each column is a cube that will convolve with a filter
    new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    #output_shape = x.shape[1] - kernel_size + 1
    #return view_as_windows(x, (kernel_size, kernel_size)).reshape(output_shape*output_shape, kernel_size*kernel_size).T

def im2col(x,hh,ww):

    """
    Args:
    x: image matrix to be translated into columns, (C,H,W)
    hh: filter height
    ww: filter width
    Returns:
    col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
    new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c,h,w = x.shape
    new_h = h - hh + 1
    new_w = w - ww+ 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
       for j in range(new_w):
           patch = x[:,i:i+hh,j:j+ww]
           col[i*new_w+j,:] = np.reshape(patch,-1)
    return col

def col2im(mul, h_prime, w_prime, C, F):
    """
    Args:
    mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
    h_prime: reshaped filter height
    w_prime: reshaped filter width
    C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    F: number of filters
    Returns:
    (F/NC,h_prime,w_prime) matrix
    """
    
    out = np.zeros([F,h_prime,w_prime])
    for i in range(F):
        col = mul[i]
        out[i] = np.reshape(col,(h_prime,w_prime))

    return out

def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
    """
    Args:
    dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
    h_prime,w_prime: height and width for the feature map
    strid: stride
    hh,ww,c: size of the filters
    Returns:
    dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c,H,W])
    for i in range(h_prime*w_prime):
        row = dim_col[i,:]
        h_start = (i / w_prime) * stride
        w_start = (i % w_prime) * stride
        dx[:,h_start:h_start+hh,w_start:w_start+ww] += np.reshape(row,(c,hh,ww))
    return dx