# Sigmoid activation function (also called the logistic functions)
# ---
# The sigmoid function has a "s-shaped" graph, and converts its output into a probability between 0 and 1,
# converting large negative numbers towards 0 and large positive number towards 1.
# For the input "0" the function returns 0.5.
# ---
# The sigmoid function is basically only used in recurrent neural networks,
# and is therefore not used in MLPs or CNNs. Instead, ReLU or leaky ReLU is used.
# The sigmoid function should only be used in the output layer when building a binary classifier,
# in which the output is interpreted as a class label depending on the probability value of
# input returned by the function.
# The sigmoid function has the vanishing gradient problem.

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    s = 1/(1 + np.exp((-x)))
    ds = 1/(1 + np.exp(x)) * (1 - 1/(1 + np.exp(x)))
    return s, ds


def sigmoid_derivative(x):
    return 1/(1 + np.exp(x)) * (1 - 1/(1 + np.exp(x)))


def visualize():
    x = np.arange(-6, 6, 0.01)
    sigmoid(x)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, sigmoid(x)[0], color="#307EC7", linewidth=3, label="sigmoid")
    ax.plot(x, sigmoid(x)[1], color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.show()

    plt.title('Sigmoid (σ)')
    plt.show()
