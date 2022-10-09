from Ejer3.neuron import Neuron

class Layer:

    def __init__(self, num_neurons, prev_num_neurons, activation, learn_rate):
        self.neurons = [Neuron(prev_num_neurons + 1, activation, learn_rate) for i in range(num_neurons)]
        self.size = num_neurons

    def get_size(self):
        return self.size

    def get_all_outputs(self):
        to_return = []
        for current_neuron in self.neurons:
            to_return.append(current_neuron.output)
        return to_return

    def get_expected_inner(self, num_neuron):
        aux = 0
        for current_neuron in self.neurons:
            aux += (current_neuron.weights[num_neuron] * current_neuron.delta)
        return aux

    def propagation(self, input):
        for i in range(0, self.size):
            self.neurons[i].calculate_output(input)

    def plot(self, num):
        print(str(num) + ": ")
        for i in range(0, self.size):
            self.neurons[i].plot()
        print("\n")
