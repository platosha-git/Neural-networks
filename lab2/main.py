import matplotlib.pyplot as plt
import os
from datetime import datetime
from dataset import get_dataset
from perceptron import *
import numpy as np


def main():
    X_train, X_test, y_train, y_test = get_dataset()

    n_x = X_train.shape[0]
    n_h = 100
    np.random.seed(7)
    epoch = 100


    tanhParams = {  'W1': np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
                    'b1': np.zeros((n_h, 1)),
                    'W2': np.random.randn(10, n_h)* np.sqrt(1. / n_h),
                    'b2': np.zeros((10, 1))
                 }

    per = Perceptron(tanhParams, tanh, dTanh)

    print('Обучение:')
    per.train(X_train, y_train, epoch)

    print('\nТестирование:')
    per.test(X_test, y_test)


if __name__ == '__main__':
    main()
