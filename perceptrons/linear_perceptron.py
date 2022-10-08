from perceptrons.perceptron import Perceptron
from perceptrons.perceptron_types import LINEAR
import numpy as np

# The expected output belongs to the real set
class LinearPerceptron(Perceptron):

    def __init__(self, training, expected_output, learning_rate):
        super().__init__(training, expected_output, learning_rate, LINEAR)

    def activation(self, excitation):
        return excitation

    def error(self, w):
        error = 0
        for i in range(len(self.training)):
            output = np.inner(self.training[i], w)
            error += (self.expected_output[i] - output) ** 2
        return 0.5 * error
