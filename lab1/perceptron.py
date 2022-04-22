from layers import *
import pandas as pd
from scipy.stats import pearsonr

def init_s_neurons(size):
    neurons = []
    for i in range(size):
        neuron = SNeuron(None, lambda value: value)
        neurons.append(neuron)
    return neurons

def init_a_neurons(size, s_size):
    neurons = []
    for i in range(size):
        neuron = ANeuron(None, lambda value: int(value >= 0), s_size)
        neurons.append(neuron)
    return neurons

def init_r_neurons(size, a_size, learning_speed):
    neurons = []
    for i in range(size):
        neuron = RNeuron(lambda: 0, lambda value: 1 if value >= 0 else 0, a_size, learning_speed, bias=0)
        neurons.append(neuron)
    return neurons


class Perceptron:
    def __init__(self, num_of_counts, s_size, a_size, learning_speed=0.01):
        self.s_neurons = init_s_neurons(s_size)
        self.a_neurons = init_a_neurons(a_size, s_size)
        self.r_neurons = init_r_neurons(num_of_counts, a_size, learning_speed)

    def solve(self, inputs):
        s_results = []
        for neuron, value in zip(self.s_neurons, inputs):
            s_results.append(neuron.solve(value))

        a_results = []
        for neuron in self.a_neurons:
            a_results.append(neuron.solve(s_results))

        r_results = []
        for neuron in self.r_neurons:
            r_results.append(neuron.solve(a_results))

        return r_results

    def correct(self, expected_results):
        for neuron, expected_result in zip(self.r_neurons, expected_results):
            neuron.correct(expected_result)

    def train(self, X_train, y_train):
        total_classifications = len(y_train) * len(y_train[0])
        min_wrong_classifications = total_classifications
        stability_time = 0
        epoch = 0
        continue_training = True

        while continue_training and stability_time < 100:
            wrong_classifications = 0
            continue_training = False
            for i in range(len(y_train)):
                results = self.solve(X_train[i])

                for result, expected_result in zip(results, y_train[i]):
                    if result != expected_result:
                        wrong_classifications += 1
                        self.correct(y_train[i])
                        continue_training = True

            epoch += 1
            print('\tЭпоха {:d}. Количество неверных классификаций: {:d}'.format(epoch, wrong_classifications))

            if min_wrong_classifications <= wrong_classifications: # начали допускать больше ошибок, после 100 таких останавливаем
                stability_time += 1
            else:
                min_wrong_classifications = wrong_classifications
                stability_time = 0

        print(
            'Обучение закончилось на эпохе {:d}\n Точность обучения: {:.1f}%'.format(
                epoch,
                float(total_classifications - min_wrong_classifications) / total_classifications * 100
            )
        )

    def del_neurons(self, to_remove, activations):
        for i in range(len(to_remove) - 1, -1, -1):
            if to_remove[i]:
                del self.a_neurons[i]
                del activations[i]
                for j in range(len(self.r_neurons)):
                    del self.r_neurons[j].input_weights[i]


    def s_solve(self, inputs):
        results = []
        for neuron, value in zip(self.s_neurons, inputs):
            results.append(neuron.solve(value))
        return results


    def optimize(self,  X_train):
        activations = [[] for _ in self.a_neurons]

        # массив из 15 массивов с 784 пикселями
        a_inputs = []
        for data in X_train:
            res = self.s_solve(data)
            a_inputs.append(res)

        # берем 784 пикселя
        for i_count, a_input in enumerate(a_inputs):
            # береи 1 нейрон из 10.000
            for n_count, neuron in enumerate(self.a_neurons):   
                # применили ф-ию активации и получили 0 или 1
                # в итоге размер activations: 10.000 по 15
                activations[n_count].append(neuron.solve(a_input))

        to_remove = [False] * len(self.a_neurons)

        len_x = len(X_train)
        print('Подсчет мертвых нейронов в слое А')
        # мертвые нейроны - те, у которых значения весов не меняются в ходе обучения
        for i, activation in enumerate(activations):
            zeros = activation.count(0)
            if zeros == 0 or zeros == len_x:
                to_remove[i] = True
        dead_neurons = to_remove.count(True)
        print('{:d}'.format(dead_neurons))

        self.del_neurons(to_remove, activations)

        corr_coef = 0.75
        to_remove = [False] * len(self.a_neurons)

        print('\nПодсчет коррелирующих нейронов в слое A')
        # коррелирующие - "дубликаты" -  пара нейронов, которые всегда выдают одинаковый результат
        for i in range(len(activations) - 1):
            if not to_remove[i]:
                for j in range(i + 1, len(activations)):
                    corr, _ = pearsonr(activations[j], activations[i])
                    if corr >= corr_coef:
                        to_remove[j] = True
                        
        correlating_neurons = to_remove.count(True) #- dead_neurons
        print('{:d}'.format(correlating_neurons))

        self.del_neurons(to_remove, activations)

        print('Удалили все мертвые и коррелирующие нейроны')
        print('Осталось {:d} нейронов'.format(len(self.a_neurons)))

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
            'Точность на тестовых данных: {:.1f}%'.format(
                float(total_classifications - misc) / total_classifications * 100
            )
        )