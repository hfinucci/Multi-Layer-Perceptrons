from cmath import exp
from random import randint
import numpy as np

# [[1,1], [-1,-1], [1,-1], [-1,1]]
class Perceptron():
    def __init__(self, training, output, learn_rate, umbral):
        self.training = training
        self.output = output
        self.learn_rate = learn_rate
        self.umbral = umbral

    # Aplica la funcion de activacion y devuelve el O
    def calculate_activation(self):
        total = 0
        for i in range(1, self.ws.length):
            total += self.ws[i] * self.training[i]
        total -= self.umbral
        return 1 if total < 0 else -1

    def calculate_error(self, expected, activation, w, u):
        total = 0
        for i in range(0, stimulus.length):
            total += abs(expected - activation)


    def delta(self, expected, calc_value, stimulus):
        return self.learn_rate * (expected - calc_value) * stimulus

    def train(self):
        i = 0
        self.w = np.zeros(self.training[0].length)
        error = 0
        error_min = 1
        while error_min > 0 and i < 1000:
            u = randint(0, self.training.length)
            activation = self.calculate_activation()
            w_array = self.delta(self.output[u], activation, np.array(self.training[u]))
            w += w_array
            error = self.calculate_error(self.output[u], activation, w, u)
            if error < error_min:
                error_min = error
                w_min = w
            i += 1
        return w_min, error

