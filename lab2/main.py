import struct
import numpy as np
import os
from dataset import get_dataset 
from multi_perceptron import MultiPerceptron

def main():
    np.random.seed(7)
    X_train, X_test, y_train, y_test = get_dataset()

    m_per = MultiPerceptron(X_train.shape[0])

    print('Обучение:')
    m_per.train(X_train, y_train, X_test, y_test)

    print('\nТестирование:')
    m_per.test(X_test, y_test)


if __name__ == '__main__':
    main()
