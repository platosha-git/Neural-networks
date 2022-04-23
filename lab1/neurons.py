import random
import numpy as np


def init_s_neurons(size):
    neurons = []
    for i in range(size):
        neuron = SNeuron(lambda value: value)
        neurons.append(neuron)
    return neurons

def init_a_neurons(size, s_size):
    neurons = []
    for i in range(size):
        neuron = ANeuron(lambda value: int(value > 0), s_size)
        neurons.append(neuron)
    return neurons

def init_r_neurons(size, a_size, learning_speed):
    neurons = []
    for i in range(size):
        neuron = RNeuron(lambda: 0, lambda value: int(value >= 0), a_size, learning_speed)
        neurons.append(neuron)
    return neurons



class SNeuron():
    def __init__(self, f_activ):
        self.activate = f_activ

    def solve(self, inputs):
        return self.activate(inputs)


class ANeuron():
    def __init__(self, f_activate, inputs_count):
        self.activate = f_activate
        self.input_weights = self.get_weights(inputs_count)
        self.bias = self.get_bias()
        self.last_inputs = None
        self.last_result = None

    def accumulate(self, inputs):
        accumulation = -self.bias
        for value, weight in zip(inputs, self.input_weights):
            accumulation += value * weight
        return accumulation

    def solve(self, inputs):
        self.last_inputs = inputs
        self.last_result = self.activate(self.accumulate(inputs))
        return self.last_result

    def get_bias(self):
        bias = 0
        for weight in self.input_weights:
            if weight > 0:
                bias += 1
            if weight < 0:
                bias -= 1
        return bias

    def get_weights(self, count):
        return [random.choice([-1, 0, 1]) for i in range(count)]


class RNeuron():
    def __init__(self, f_initialize, f_activate, inputs_count, learning_speed, bias=0):
        self.initialize = f_initialize
        self.activate = f_activate
        self.input_weights = self.get_weights(inputs_count)
        self.last_inputs = None
        self.last_result = None
        self.learning_speed = learning_speed
        self.bias = bias

    def accumulate(self, inputs):
        accumulation = - self.bias
        for value, weight in zip(inputs, self.input_weights):
            accumulation += value * weight
        return accumulation

    def solve(self, inputs):
        self.last_inputs = inputs
        self.last_result = self.activate(self.accumulate(inputs))
        return self.last_result

    def get_weights(self, count):
        return [self.initialize() for _ in range(count)]

    def correct(self, expected_result):
        if expected_result != self.last_result:
            lr = 1
            if self.last_result == 0:
                lr = -1
            delta_weight = [lr * last_input for last_input in self.last_inputs]
            
            self.input_weights = [
                input_weight - delta_weight * self.learning_speed
                for input_weight, delta_weight in zip(self.input_weights, delta_weight)
            ]
            self.bias += lr * self.learning_speed
            