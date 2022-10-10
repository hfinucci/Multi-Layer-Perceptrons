from Ejer3.multi_perceptron import MultiPerceptron
import numpy as np
from Ejer3.activation import *

training_arr = []

with open('Ejer3/TP2-ej3-digitos.txt', 'r') as txtfile:
    line_count = 0
    aux_array = []
    for line in txtfile:
        aux_array.append(int(line[0]))
        aux_array.append(int(line[2]))
        aux_array.append(int(line[4]))
        aux_array.append(int(line[6]))
        aux_array.append(int(line[8]))
        line_count += 1
        if line_count % 7 == 0:
            training_arr.append(aux_array)
            aux_array = []
txtfile.close()

activation = Sigmoid()

outputs = [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]

multi_layer = MultiPerceptron([35, 10, 5, 1], 1, activation)
#multi_layer.plot()

multi_layer.train(training_arr, outputs, 100)


print("========================")
print(multi_layer.test([training_arr[1],
                        training_arr[6],
                        training_arr[2],
                        training_arr[7],
                        training_arr[3],
                        training_arr[0],
                        training_arr[8],
                        training_arr[4],
                        training_arr[9],
                        training_arr[5],
                        ]))
