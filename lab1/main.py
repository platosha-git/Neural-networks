from dataset import get_dataset, NUM_OF_COUNTS, SIZE_OF_ONE_IMAGE
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

CORR_COEF = 0.75

def main():
    labels, images = get_dataset()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=15, train_size=25)
    
    per = Perceptron(SIZE_OF_ONE_IMAGE, 500, NUM_OF_COUNTS)
    
    print('Обучение:')
    per.train(X_train, y_train)
    
    print('\nОптимизация:')
    per.optimize(X_train, CORR_COEF)
    
    per.test(X_test, y_test)


if __name__ == '__main__':
    main()
