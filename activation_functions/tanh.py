# Tanh activation function (tangent hyperbolic)
# ---
# The output of the tanh activation function, will always range between -1 and 1.
# The tanh function is zero-centered, which makes the optimization process muh easier than sigmoid,
# and has a steeper gradient than the sigmoid function.
# ---
# The tanh function is primarily used in recurrent neural networks, and is no usually used in
# MLPs and CNNs. Instead, ReLU og leaky ReLU is used.
# Tanh should never be used in the output layer.
# Tanh has the vanishing gradient problem.
# Tanh is always preferred over the sigmoid activation function.

import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    t = (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))
    dt = 1 - t ** 2
    return t, dt


def tanh_derivative(x):
    return 1 - tanh(x)[0] ** 2


def visualize():
    z = np.arange(-4, 4, 0.01)
    tanh(z)[0].size,tanh(z)[1].size

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(z, tanh(z)[0], color="#307EC7", linewidth=3, label="tanh")
    ax.plot(z, tanh(z)[1], color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('Tanh (tangent hyperbolic)')
    fig.show()
