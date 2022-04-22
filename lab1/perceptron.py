from layers import *
import pandas as pd
from scipy.stats import pearsonr

class Perceptron:
    def __init__(self, numbers_count, inputs_count, learning_speed=0.01, a_neurons_count=5000):
        print('Генерируем слой S (сенсорный)')
        self.s_layer = SNeuronLayer(inputs_count)
        print('Генерируем слой A (ассоциативный)')
        self.a_layer = ANeuronLayer(a_neurons_count, inputs_count)
        print('Генерируем слой R (реагирующий)')
        self.r_layer = RNeuronLayer(numbers_count, a_neurons_count, learning_speed)

    def solve(self, inputs):
        s_result = self.s_layer.solve(inputs)
        a_result = self.a_layer.solve(s_result)
        return self.r_layer.solve(a_result)

    def correct(self, expected_results):
        self.r_layer.correct(expected_results)

    def train(self, X_train, y_train):
        assert len(X_train) == len(y_train)
        print('\n Обучение')

        continue_training = True
        epoch = 0

        total_classifications = len(y_train) * len(y_train[0])
        min_wrong_classifications = total_classifications
        stability_time = 0
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
            print('"Закончилась эпоха {:d}. Количество неверных классификаций: {:d}'.format(epoch, wrong_classifications))

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
                del self.a_layer.neurons[i]
                del activations[i]
                for j in range(len(self.r_layer.neurons)):
                    del self.r_layer.neurons[j].input_weights[i]


    # удаление мертвых и коррелирующих нейронов
    def optimize(self,  X_train):
        print('\nОптимизация')
        # 10.000 массивов
        activations = [[] for _ in self.a_layer.neurons]

        # массив из 15 массивов с 784 пикселями
        a_inputs = [self.s_layer.solve(data) for data in X_train]

        # берем 784 пикселя
        for i_count, a_input in enumerate(a_inputs):
            # береи 1 нейрон из 10.000
            for n_count, neuron in enumerate(self.a_layer.neurons):   
                # применили ф-ию активации и получили 0 или 1
                # в итоге размер activations: 10.000 по 15
                activations[n_count].append(neuron.solve(a_input))

        to_remove = [False] * len(self.a_layer.neurons)

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
        to_remove = [False] * len(self.a_layer.neurons)

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
        print('Осталось {:d} нейронов'.format(len(self.a_layer.neurons)))