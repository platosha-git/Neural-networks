from neurons import *


class NeuronLayer:
    def __init__(self):
        self.neurons = []

    def solve(self, inputs):
        results = []
        for neuron in self.neurons:
            results.append(neuron.solve(inputs))
        return results

    def correct(self, expected_results):
        pass


class SNeuronLayer(NeuronLayer):
    def __init__(self, neurons_count):
        super().__init__()
        for _ in range(neurons_count):
            self.add_neuron(None, lambda value: value)

    def add_neuron(self, f_initialize, f_activate):
        neuron = SNeuron(f_initialize, f_activate)
        self.neurons.append(neuron)

    def solve(self, inputs):
        results = []
        for neuron, value in zip(self.neurons, inputs):
            results.append(neuron.solve(value))
        return results


class ANeuronLayer(NeuronLayer):
    def __init__(self, neurons_count, inputs_count):
        super().__init__()
        for _ in range(neurons_count):
            self.add_neuron(inputs_count, None, lambda value: int(value >= 0))

    def add_neuron(self, inputs_count, f_initialize, f_activate):
        neuron = ANeuron(f_initialize, f_activate, inputs_count)
        self.neurons.append(neuron)


class RNeuronLayer(NeuronLayer):
    def __init__(self, neurons_count, a_neurons_count, learning_speed):
        super().__init__()
        for _ in range(neurons_count):
            self.add_neuron(lambda: 0,
                            lambda value: 1 if value >= 0 else 0,
                            a_neurons_count,
                            learning_speed,
                            bias=0)

    def add_neuron(self, f_initialize, f_activate, inputs_count, learning_speed, bias):
        neuron = RNeuron(f_initialize, f_activate, inputs_count, learning_speed, bias)
        self.neurons.append(neuron)

    def correct(self, expected_results):
        for neuron, expected_result in zip(self.neurons, expected_results):
            neuron.correct(expected_result)
