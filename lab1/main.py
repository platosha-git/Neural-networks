from dataset import get_dataset, NUM_OF_COUNTS, SIZE_OF_ONE_IMAGE
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

import matplotlib as plt

def train_perceptron(X_train, y_train):
    input_count = SIZE_OF_ONE_IMAGE
    network = Perceptron(NUM_OF_COUNTS, input_count, a_neurons_count=1000)
    network.train(X_train, y_train)
    network.optimize(X_train)
    return network

def test_network(network, X_test, y_test):
    assert len(X_test) == len(y_test)
    total_classifications = len(y_test) * len(y_test[0])
    misc = 0
    for i in range(len(X_test)):
        results = network.solve(X_test[i])
        for result, expected_result in zip(results, y_test[i]):
            if result != expected_result:
                misc += 1

    print(
        'Точность на тестовых данных: {:.1f}%'.format(
            float(total_classifications - misc) / total_classifications * 100
        )
    )


def main():
    labels, images = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=15, train_size=25)
    network = train_perceptron(X_train, y_train)
    test_network(network, X_test, y_test)


if __name__ == '__main__':
    main()
