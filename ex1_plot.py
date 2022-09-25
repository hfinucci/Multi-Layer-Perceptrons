from matplotlib import pyplot as plt
import numpy as np

# Documentation of pyplot
# https://matplotlib.org/stable/api/pyplot_summary.html?highlight=pyplot

# TODO: Transform function to use all weights obtained from training
def plot_graph(points, output, weight):
    fig, ax = plt.subplots()

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Step Perceptron Graph')

    for i in range(len(points)):
        if output[i] == 1:
            ax.scatter(points[i][0], points[i][1], color='b')
        else:
            ax.scatter(points[i][0], points[i][1], color='r')
    
    min_lim = -2
    max_lim = 2

    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)

    x = np.linspace(-2, 2, 100)
    y = -((weight[0] * x + weight[2]) / weight[1])

    ax.plot(x, y, color="black")
    plt.show()