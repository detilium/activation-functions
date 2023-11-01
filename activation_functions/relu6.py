import numpy as np
import matplotlib.pyplot as plt


def relu6(x):
    r = [min(value, 6) if value > 0 else 0.0 * value for value in x]
    dr = [1 if 0 < value <= 6 else 0.0 for value in x]
    return r, dr


def relu6_derivative(x):
    return [1 if 0 < value <= 6 else 0.0 for value in x]


def visualize():
    x = np.arange(-10, 10, 0.05)
    relu6(x)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, relu6(x)[0], color="#307EC7", linewidth=3, label="relu6")
    ax.plot(x, relu6(x)[1], color="#9621E2", linewidth=3, label="derivative")
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('ReLU6')
    fig.show()
