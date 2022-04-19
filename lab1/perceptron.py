from layers import *


class Perceptron:
    def __init__(self, numbers_count, inputs_count, learning_speed=0.01, a_neurons_count=500):
        print('Генерация S-элементов...')
        self.s_layer = SNeuronLayer(inputs_count)

        print('Генерация A-элементов...')
        self.a_layer = ANeuronLayer(a_neurons_count, inputs_count)

        print('Генерация R-элемента...')
        self.r_layer = RNeuronLayer(numbers_count, a_neurons_count, learning_speed)

    def solve(self, inputs):
        s_result = self.s_layer.solve(inputs)
        a_result = self.a_layer.solve(s_result)
        return self.r_layer.solve(a_result)

    def correct(self, expected_results):
        self.r_layer.correct(expected_results)

    def train(self, X_train, y_train):
        assert len(X_train) == len(y_train)
        print('\nОбучение:')

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
            print('\tЭпоха {:d}. {:d} неверных классификаций'.format(epoch, wrong_classifications))

            if min_wrong_classifications <= wrong_classifications:
                stability_time += 1
            else:
                min_wrong_classifications = wrong_classifications
                stability_time = 0

        print(
            'Точность обучения: {:.1f}%'.format(
                float(total_classifications - min_wrong_classifications) / total_classifications * 100
            )
        )

    # удаление мертвых и коррелирующих нейронов
    def optimize(self,  X_train):
        print('\nОптимизация:')

        activations = [[] for _ in self.a_layer.neurons]
        a_inputs = [self.s_layer.solve(data) for data in X_train]
        
        for i_count, a_input in enumerate(a_inputs):
            for n_count, neuron in enumerate(self.a_layer.neurons):
                activations[n_count].append(neuron.solve(a_input))
        to_remove = [False] * len(self.a_layer.neurons)

        a_layer_size = len(self.a_layer.neurons)
        for i, activation in enumerate(activations):
            zeros = activation.count(0)
            if zeros == 0 or zeros == a_layer_size:
                to_remove[i] = True

        dead_neurons = to_remove.count(True)
        print('\t{:d} мертвых нейронов в слое А'.format(dead_neurons))


        for i in range(len(activations) - 1):
            if not to_remove[i]:
                for j in range(i + 1, len(activations)):
                    if activations[j] == activations[i]:
                        to_remove[j] = True
        
        correlating_neurons = to_remove.count(True) - dead_neurons
        print('\t{:d} коррелирующих нейронов в слое A'.format(correlating_neurons))

        for i in range(len(to_remove) - 1, -1, -1):
            if to_remove[i]:
                del self.a_layer.neurons[i]
                for j in range(len(self.r_layer.neurons)):
                    del self.r_layer.neurons[j].input_weights[i]

        print('Удалили все мертвые и коррелирующие нейроны')
        print('Осталось {:d} нейронов'.format(len(self.a_layer.neurons)))
