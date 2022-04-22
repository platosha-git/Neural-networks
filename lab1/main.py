from dataset import get_dataset, NUM_OF_COUNTS, SIZE_OF_ONE_IMAGE
from sklearn.model_selection import train_test_split
from perceptron import Perceptron


def main():
    labels, images = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=15, train_size=25)
    
    network = Perceptron(NUM_OF_COUNTS, SIZE_OF_ONE_IMAGE, a_neurons_count=1000)
    
    network.train(X_train, y_train)
    network.optimize(X_train)
    network.test(X_test, y_test)


if __name__ == '__main__':
    main()
