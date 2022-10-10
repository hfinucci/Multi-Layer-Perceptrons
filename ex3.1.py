from ast import And
from Ejer3.multi_perceptron import MultiPerceptron
import json
from Ejer3.constants import *
from Ejer3.activation import *

with open("Ejer3/ex3.1_config.json") as file:
    jsonObject = json.load(file)
    file.close()

learning_rate = float(jsonObject["learning_rate"])
generation = int(jsonObject["generation"])
operation = str(jsonObject["operation"])

training_set = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])

if operation == AND:
    expected_output = np.array([[-1], [-1], [-1], [1]])
else:
    expected_output = np.array([[1], [1], [-1], [-1]])

activation = Sigmoid()
multi_layer = MultiPerceptron([2, 2, 1], learning_rate, activation)
error = multi_layer.train(training_set, expected_output, generation)
print(multi_layer.test(training_set))
#plot_graph(training_set, expected_output, min_w)
#plot_errors(errors)

# results = perceptron.test(np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]]))
# print(results)
# print("Min weight: ", min_w)
# multi_layer.save('Ejer3/weights.txt')
