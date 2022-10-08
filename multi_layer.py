
import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        if -700 < x < 700:
            return np.exp(x) / (1 + np.exp(x))
        return 0 if x < 0 else 1

    @staticmethod
    def sigmoid_dx(x):
        # se hace 0 despues de este valor
        if -355 < x < 355:
            return np.exp(x) / np.power(np.exp(x) + 1, 2)
        return 0

    @staticmethod
    def tanh(excitation):
        return np.tanh(excitation)

    @staticmethod
    def tanh_dx(excitation):
        return 1 - np.tanh(excitation) ** 2


class Neuron:
    def __init__(self, weight_num, Activation, learn_rate ):
        # self.weights = np.array(ndim=weight_num)
        self.weights = np.random.uniform(-1, 1, weight_num)
        # self.weights = np.random.rand(weight_num,0)
        self.delta = 0
        self.output = 0 
        self.activation = Activation
        self.learn_rate = learn_rate

    
    def calculate_output(self, prev_stimulus):
        E = np.inner(prev_stimulus,  self.weights)
        self.output =  self.activation(E)
    
    def calculate_delta(self, expected):
        pass        

class Layer:
    

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = [Neuron(prev_num_neurons + 1, activation, learn_rate) for i in range(num_neurons)]
        self.num_neurons = num_neurons

        print(self.neurons)
        for i in self.neurons:
            print(i.weights)
        print('----')


class MultiPerceptron:
    
    ERROR_MIN = 0.001
    def __init__(self, net_congif, learn_rate, Activation):
        
        # self.matrix = np.empty([len(net_congif)-1,0])
        self.matrix = []
        for i in range(1, len(net_congif)):
            self.matrix.append(Layer( net_congif[i], net_congif[i-1], Activation, learn_rate))

    def train(self, training, expected_output, max_gen):
        current_gen = 0
        error = 0
        self.error_min = np.inf
        while self.error_min > self.ERROR_MIN and current_gen < max_gen:
            #forward_propagation()
            #error_min = calculate_error()
            #back_propagation()
            pass

    
    def test(self, test_set):
        pass
    