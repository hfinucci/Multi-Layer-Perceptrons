from Ejer3.multi_perceptron import MultiPerceptron


multi_layer = MultiPerceptron([3,2,1], 1, 0)

multi_layer.train([], [], 100)

# multi_layer.save('Ejer3/weights.txt')

# with open('Ejer3/TP2-ej3-digitos.txt', 'r') as txtfile:
#     for line in txtfile:
#         print(line)
