import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp((-x)))


def swish(x):
    return x * sigmoid(x)
    # or
    # return x * (1 / (1 + np.exp((-x))))


def swish_derivative(x):
    return swish(x) + sigmoid(x) * (1 - swish(x))
    # or
    # return swish(x) + (1 / (1 + np.exp((-x)))) * (1 - swish(x))


def visualize():
    x = np.arange(-10, 10, 0.1)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, swish(x), color="#307EC7", linewidth=3, label="swish")
    ax.plot(x, swish_derivative(x), color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('Swish')
    fig.show()
