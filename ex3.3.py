# from ast import And
# from Ejer3.multi_perceptron import MultiPerceptron
# import json
# from Ejer3.constants import *
# from Ejer3.activation import *
#
# with open("Ejer3/ex3.3_config.json") as file:
#     jsonObject = json.load(file)
#     file.close()
#
# learning_rate = float(jsonObject["learning_rate"])
# generation = int(jsonObject["generation"])
# operation = str(jsonObject["operation"])
#
# training_set = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
#
#
# if operation == "tahn":
#     activation = Tanh()
# else:
#     activation = Sigmoid()
#
#
#
#
#
#
#
#
#
# multi_layer = MultiPerceptron([2, 2, 1], learning_rate, activation)
#
# def parse(file_name: str, normalize= None):
#     df = pd.read_csv(file_name, sep=' +', engine='python', header=None)
#     max_num = df.values.max()
#     min_num = df.values.min()
#     if normalize:
#         if normalize == "TANH":
#             for i in range(len(df.values)):
#                 df.values[i] = 2 * (df.values[i] - min_num) / (max_num - min_num) - 1
#         elif normalize == "SIGMOID":
#             for i in range(len(df.values)):
#                 df.values[i] = (df.values[i] - min_num) / (max_num - min_num)
#
#     return df.to_numpy(), max_num , min_num
#
#
#
#
#
#
# error = multi_layer.train(training_set, expected_output, generation)
# print('--------------------\n')
# #print(multi_layer)
# print('--------------------\n')
# print(multi_layer.test(training_set))
#
#
#
#
#
# #plot_graph(training_set, expected_output, min_w)
# #plot_errors(errors)
#
# # results = perceptron.test(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]))
# # print(results)
# # print("Min weight: ", min_w)
# # multi_layer.save('Ejer3/weights.txt')


from multiperceptron.multi_perceptron import MultiPerceptron
from multiperceptron.activation import *
import numpy as np

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

outputs = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

multi_layer = MultiPerceptron([35, 20, 15, 10], 0.01, activation)
# multi_layer.plot()

multi_layer.train(training_arr, outputs, 1000000, 0.001, 10 )

print("========================")

result = multi_layer.test([training_arr[1],
                           training_arr[6],
                           training_arr[2],
                           training_arr[7],
                           training_arr[3],
                           training_arr[0],
                           training_arr[8],
                           training_arr[4],
                           training_arr[9],
                           training_arr[5]
                           ])
print(result[0])

to_print = []
for data in result:
    to_print.append(data.index(max(data)))

print(to_print)
