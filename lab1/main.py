from sklearn.model_selection import train_test_split

from perceptron import Perceptron
from training_dataset import NUMBER_COUNT, read_dataset, SIZE_OF_ONE_IMAGE

def train_perceptron(X_train, y_train):
    input_count = SIZE_OF_ONE_IMAGE
    network = Perceptron(NUMBER_COUNT, input_count, a_neurons_count=10000)
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
        '\nТочность на тестовых данных: {:.2f}%'.format(
            float(total_classifications - misc) / total_classifications * 100
        )
    )


def main():
    images, labels, = read_dataset()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=10, train_size=15)
    network = train_perceptron(X_train, y_train)
    test_network(network, X_test, y_test)


if __name__ == '__main__':
    main()