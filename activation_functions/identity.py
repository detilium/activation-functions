import numpy as np
import matplotlib.pyplot as plt


def identity(x):
    return x


def identity_derivative(x):
    return x * 1


def visualize():
    x = np.arange(-5, 5, 0.01)
    identity(x)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, identity(x), color="#307EC7", linewidth=3, label="identity")
    ax.plot(x, identity_derivative(x), color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('Identity')
    fig.show()
