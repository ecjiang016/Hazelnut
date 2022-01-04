from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def display(fig, cnn_layout, cost_list):
    """
    Displays all the filters of a CNN. Layers seperated row-wise.
    Filters in layers seperated column-wise. If kernel is 3D, is is displayed with channels merged horizontally.

    Inputs:
    fig: fig = plt.figure(constrained_layout=True) *Pass fig only once to prevent multiple windows
    cnn_layout: Layout of the convolutional neural network
    cost_list: List of all the cost values to plot
    """
    kernels = []
    for module in cnn_layout:
        if module[0] == "Kernel":
            kernels.append(module[1])

    layers = len(kernels)

    gs = GridSpec(layers+(layers//2 +1), max([kernel.shape[0] for kernel in kernels]), figure=fig)
    
    plt.clf()
    # plot first few filters
    for i in range(layers):
        for j in range(kernels[i].shape[0]):
            # specify subplot and turn of axis
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            kernel = np.hstack([channel for channel in kernels[i][j]])
            ax.imshow(kernel, cmap='gray')

    # show the figure
    ax = plt.subplot(gs[layers:, :])
    ax.plot(cost_list)
    plt.draw()
    plt.pause(0.1)
