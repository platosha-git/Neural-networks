import random


class Neuron:
    def __init__(self, f_initialize, f_activate):
        self.initialize = f_initialize
        self.activate = f_activate

    def solve(self, inputs):
        raise NotImplementedError

    def correct(self, expected_result):
        pass


class SNeuron(Neuron):
    def solve(self, inputs):
        return self.activate(inputs)


class ActivationNeuron(Neuron):
    def __init__(self, f_initialize, f_activate, inputs_count):
        super().__init__(f_initialize, f_activate)
        self.input_weights = self.get_weights(inputs_count)
        self.bias = self.get_bias()
        self.last_inputs = None
        self.last_result = None

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

    def get_bias(self):
        return self.initialize()


class ANeuron(ActivationNeuron):
    def get_bias(self):
        bias = 0
        for weight in self.input_weights:
            if weight > 0:
                bias += 1
            if weight < 0:
                bias -= 1
        return bias

    def get_weights(self, count):
        return [random.choice([-1, 0, 1]) for _ in range(count)]


class RNeuron(ActivationNeuron):
    def __init__(self, f_initialize, f_activate, inputs_count, learning_speed, bias):
        super().__init__(f_initialize, f_activate, inputs_count)
        self.learning_speed = learning_speed
        self.bias = bias

    # корректировка весов при ошибке
    def correct(self, expected_result):
        if expected_result != self.last_result:
            # градиентный спуск
            lr = 1
            if self.last_result == 0:
                lr = -1
            delta_weight = [lr * last_input for last_input in self.last_inputs]
            self.input_weights = [
                input_weight - delta_weight * self.learning_speed
                for input_weight, delta_weight in zip(self.input_weights, delta_weight)
            ]
            self.bias += lr * self.learning_speed