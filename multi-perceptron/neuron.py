import numpy as np
from activacion import *


class Neuron:
    def __init__(self, weight_num, activation: Activation, learn_rate):
        self.weights = np.random.uniform(-1, 1, weight_num + 1)
        self.output = 0
        self.error = 0
        self.output_dx = 0
        self.activation = activation
        self.learn_rate = learn_rate

    def calculate_output(self, inputs: np.ndarray):
        inputs = np.append(inputs, 1)
        e = np.inner(inputs, self.weights)
        self.output = self.activation.apply(e)
        self.output_dx = self.activation.apply_dx(e)

    def calculate_error(self, delta: np.ndarray):
        self.error = self.output_dx * delta

    def update_w(self, inputs: np.ndarray):
        inputs = np.append(inputs, 1)
        for i in range(0, len(self.weights)):
            self.weights[i] += self.learn_rate * self.error * inputs[i]

