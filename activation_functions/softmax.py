import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def softmax_derivative(Y):
    s = Y.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def visualize():
    x = np.arange(0, 10, 0.5)
    softmax(x)

    # Setup centered axes
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Create and show plot
    ax.plot(x, softmax(x), color="#307EC7", linewidth=3, label="softmax")
    # Softmax derivative is not really visualizable, and lags context without the previous layer in a neural network
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle('ReLU')
    fig.show()
