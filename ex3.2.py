from Ejer3.multi_perceptron import MultiPerceptron
import numpy as np

training_arr = []
expected_output = []

with open('Ejer3/TP2-ej3-digitos.txt', 'r') as txtfile:
    for line in txtfile:
        inner_arr = [
            int(line[0]),
            int(line[2]),
            int(line[4]),
            int(line[6]),
            int(line[8])
        ]
        training_arr.append(inner_arr)
        aux = []
        aux.append(int(line[8]))
        expected_output.append(aux)

multi_layer = MultiPerceptron([3, 2, 1], 1, 0)

multi_layer.train(training_arr, expected_output, 100)