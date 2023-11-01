import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    r = [value if value > 0 else 0.0 for value in x]
    dr = [1 if value > 0 else 0.0 for value in x]
    return r, dr


def rely_derivative(x):
    return [1 if value > 0 else 0.0 for value in x]


def visualize():
    x = np.arange(-4, 4, 0.01)
    relu(x)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, relu(x)[0], color="#307EC7", linewidth=3, label="leaky relu")
    ax.plot(x, relu(x)[1], color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('ReLU')
    fig.show()