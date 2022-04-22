from neurons import *

class SLayer():
    def __init__(self, size):
        self.neurons = []
        for i in range(size):
            neuron = SNeuron(None, lambda value: value)
            self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron, value in zip(self.neurons, inputs):
            results.append(neuron.solve(value))
        return results


class ALayer():
    def __init__(self, size, s_size):
        self.neurons = []
        for i in range(size):
            neuron = ANeuron(None, lambda value: int(value >= 0), s_size)
            self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results


class RLayer():
    def __init__(self, size, a_size, learning_speed):
        self.neurons = []
        for i in range(size):
            neuron = RNeuron(lambda: 0, lambda value: 1 if value >= 0 else 0, a_size, learning_speed, bias=0)
            self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results

    def correct(self, expected_results):
        for neuron, expected_result in zip(self.neurons, expected_results):
            neuron.correct(expected_result)
