import matplotlib.pyplot as plt
import os
from datetime import datetime
from dataset import get_dataset
import numpy as np


def softMax(X):
    e = np.exp(X)
    p = e / np.sum(e, axis=0)
    return p


def ReLU(z):
    return np.maximum(0,z)


def sigmoid(z):
    return 1./(1. + np.exp(-z))


def tanh(z):
    return np.tanh(z)

def dReLU(z):
    return (z > 0) * 1

def dSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

def dTanh(z):
    return 1 / (np.cosh(z) ** 2)


def forward(X, params, activation):
    forwardPass = {}
    forwardPass['Z1'] = np.matmul(params['W1'], X) + params['b1']
    forwardPass['A1'] = activation(forwardPass['Z1'])
    
    forwardPass['Z2'] = np.matmul(params['W2'],forwardPass['A1']) + params['b2']
    forwardPass['A2'] = softMax(forwardPass['Z2'])
    
    return forwardPass


def back(X, y, forwardPass, params, dActivation):
    m = X.shape[1]
    gradient = {}
    
    gradient['dZ2'] = forwardPass['A2'] - y
    gradient['dW2'] = (1./m) * np.matmul(gradient['dZ2'], forwardPass['A1'].T)
    gradient['db2'] = (1./m) * np.sum(gradient['dZ2'], axis=1, keepdims=True)
    gradient['dA1'] = np.matmul(params['W2'].T, gradient['dZ2'])
    
    gradient['dZ1'] = gradient['dA1'] * dActivation(forwardPass['Z1'])
    gradient['dW1'] = (1./m) * np.matmul(gradient['dZ1'], X.T)
    gradient['db1'] = (1./m) * np.sum(gradient['dZ1'])
    
    return gradient


def updater(params, grad, eta, lamda, m):
    updatedParams = {}
    updatedParams['W2'] = params['W2'] - eta * grad['dW2']
    updatedParams['b2'] = params['b2'] - eta * grad['db2']
    
    updatedParams['W1'] = params['W1'] - eta * grad['dW1']
    updatedParams['b1'] = params['b1'] - eta * grad['db1']
    
    return updatedParams


def classifer(X, params, activation):
    Z1 = np.matmul(params['W1'], X) + params['b1']
    A1 = activation(Z1)

    Z2 = np.matmul(params['W2'], A1) + params['b2']
    A2 = softMax(Z2)

    pred = np.argmax(A2, axis=0)
    return pred


def main():
    X_train, X_test, y_train, y_test = get_dataset()


    m = 10000
    n_x = X_train.shape[0]
    n_h = 100
    eta = 1
    lamda = 2
    np.random.seed(7)
    epoch = 300


    tanhParams = {  'W1': np.random.randn(n_h, n_x)* np.sqrt(1. / n_x),
                    'b1': np.zeros((n_h, 1)),
                    'W2': np.random.randn(10, n_h)* np.sqrt(1. / n_h),
                    'b2': np.zeros((10, 1))
                }

    print('Обучение:')
    for i in range(epoch):
        idx = np.random.permutation(X_train.shape[1])[:m]
        X=X_train[:,idx]
        y=y_train[:,idx]
        
        forwardPass = forward(X,tanhParams,tanh)

        gradient = back(X, y, forwardPass, tanhParams,dTanh)
        tanhParams=updater(tanhParams,gradient,eta,lamda,m)

        if i % 10 == 0:
            print(f'\tЭпоха {i}.')


    y_hat = classifer(X_test, tanhParams, tanh)

    print(
            '\nТочность на тестовых данных: {:.2f}%'.format(
                sum(y_hat==y_test) * 1 / len(y_test) * 100
            )
        )


if __name__ == '__main__':
    main()
