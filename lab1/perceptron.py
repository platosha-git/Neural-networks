from neurons import *
import pandas as pd
from scipy.stats import pearsonr


class Perceptron:
    def __init__(self, s_size, a_size, r_size, learning_speed=0.01):
        self.s_neurons = init_s_neurons(s_size)
        self.a_neurons = init_a_neurons(a_size, s_size)
        self.r_neurons = init_r_neurons(r_size, a_size, learning_speed)

#TRAIN
    def s_solve(self, inputs):
        result = []
        for neuron, value in zip(self.s_neurons, inputs):
            result.append(neuron.solve(value))
        return result

    def ar_solve(self, neurons, inputs):
        result = []
        for neuron in neurons:
            result.append(neuron.solve(inputs))
        return result

    def solve(self, inputs):
        s_results = self.s_solve(inputs)
        a_results = self.ar_solve(self.a_neurons, s_results)
        r_results = self.ar_solve(self.r_neurons, a_results)
        return r_results


    def correct(self, expected_results):
        for neuron, expected_result in zip(self.r_neurons, expected_results):
            neuron.correct(expected_result)


    def train(self, X_train, y_train):
        epoch = 0
        mistackes = -1

        while (mistackes != 0):
            mistackes = 0

            for i in range(len(y_train)):
                results = self.solve(X_train[i])

                for result, expected_result in zip(results, y_train[i]):
                    if result != expected_result:
                        mistackes += 1
                        self.correct(y_train[i])

            epoch += 1
            print('\tЭпоха ' + str(epoch) + '. Ошибок: ' + str(mistackes))


#OPTIMIZE
    def get_all_neurons(self, X_train):
        all_neurons = [[] for neuron in self.a_neurons]
        a_inputs = [self.s_solve(x) for x in X_train]

        for a_input in a_inputs:
            for idx, neuron in enumerate(self.a_neurons):   
                all_neurons[idx].append(neuron.solve(a_input))

        return all_neurons

    def get_dead_neurons(self, all_neurons, len_x):
        dead_neurons = [False] * len(self.a_neurons)

        for i, neuron in enumerate(all_neurons):
            zeros = neuron.count(0)
            if zeros == 0 or zeros == len_x:
                dead_neurons[i] = True
        return dead_neurons

    def get_corr_neurons(self, all_neurons, corr_coef):
        corr_neurons = [False] * len(self.a_neurons)

        for i in range(len(all_neurons) - 1):
            for j in range(i + 1, len(all_neurons)):
                corr, _ = pearsonr(all_neurons[i], all_neurons[j])
                if corr >= corr_coef:
                    corr_neurons[j] = True
        return corr_neurons

    def remove_neurons(self, bad_neurons, all_neurons):
        for i in range(len(bad_neurons) - 1, -1, -1):
            if bad_neurons[i]:
                del self.a_neurons[i]
                del all_neurons[i]
                for j in range(len(self.r_neurons)):
                    del self.r_neurons[j].input_weights[i]

    def optimize(self, X_train, corr_coef):
        all_neurons = self.get_all_neurons(X_train)

        dead_neurons = self.get_dead_neurons(all_neurons, len(X_train))
        num_dead_neurons = dead_neurons.count(True)
        self.remove_neurons(dead_neurons, all_neurons)
        print('\tМертвых нейронов: ' + str(num_dead_neurons))

        corr_neurons = self.get_corr_neurons(all_neurons, corr_coef)        
        num_corr_neurons = corr_neurons.count(True)
        self.remove_neurons(corr_neurons, all_neurons)
        print('\tКоррелирующих нейронов: ' + str(num_corr_neurons))

        print('\tОставшихся нейронов: ' + str(len(self.a_neurons)))


    def test(self, X_test, y_test):
        assert len(X_test) == len(y_test)
        total_classifications = len(y_test) * len(y_test[0])
        misc = 0
        for i in range(len(X_test)):
            results = self.solve(X_test[i])
            for result, expected_result in zip(results, y_test[i]):
                if result != expected_result:
                    misc += 1

        print(
            '\nТочность на тестовых данных: {:.1f}%'.format(
                float(total_classifications - misc) / total_classifications * 100
            )
        )
