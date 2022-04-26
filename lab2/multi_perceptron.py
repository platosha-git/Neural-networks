import numpy as np
from base import *

class MultiPerceptron():
    def __init__(self, s_size):
        self.m = M

        self.s_size = s_size
        self.a_size = A_SIZE
        self.r_size = R_SIZE

        self.eta = 1
        self.epoch = 300

        self.params = define_sigParams(self.s_size, self.a_size, self.r_size)


#TRAIN
    def forward(self, X):
        forwardPass = {}
        forwardPass['Z1'] = np.matmul(self.params['W1'], X) + self.params['b1']
        forwardPass['A1'] = sigmoid(forwardPass['Z1'])
        forwardPass['Z2'] = np.matmul(self.params['W2'], forwardPass['A1']) + self.params['b2']
        forwardPass['A2'] = sigmoid(forwardPass['Z2'])     
        return forwardPass

    def back(self, X, y, forwardPass, epoch):
        m = X.shape[1]
        gradient = {}

        #Ошибка Е
        gradient['dZ2'] = (forwardPass['A2'] - y)
        delta = gradient['dZ2'] * dSigmoid(forwardPass['Z2'])

        gradient['dW2'] =  (1. / m) * np.matmul(delta, forwardPass['A1'].T)
        gradient['db2'] =  (1. / m) * np.sum(delta, axis=1, keepdims=True) 
        
        gradient['dA1'] =  np.matmul(self.params['W2'].T, delta)
        if (epoch == 290):
            print('Ошибка: ', gradient['dA1'])

        gradient['dZ1'] = gradient['dA1'] * dSigmoid(forwardPass['Z1'])

        gradient['dW1'] =(1. / m) * np.matmul(gradient['dZ1'], X.T)
        gradient['db1'] =(1. / m) * np.sum(gradient['dZ1'])

        return gradient

    def updater(self, grad):
        self.params['W2'] -= self.eta * grad['dW2']
        self.params['b2'] -= self.eta * grad['db2']
        self.params['W1'] -= self.eta * grad['dW1']
        self.params['b1'] -= self.eta * grad['db1']

    def train(self, X_train, y_train, X_test, y_test):
        stab_time = 0
        error_pred = 10**9
        i = 0
        while stab_time < 10 and i < self.epoch:
            idx = np.random.permutation(X_train.shape[1])[:self.m]
            X = X_train[:, idx]
            y = y_train[:, idx]
            
            forwardPass = self.forward(X)
            gradient = self.back(X, y, forwardPass, i)
            self.updater(gradient)

            if i % 10 == 0:
                print("\tЭпоха " + str(i))
            
            i += 1
            if i > 100:
                y_net_test = self.forward(X_test)['A2']
                error = np.sum(np.abs(y_net_test - y_test))
                
                if error >= error_pred:
                    stab_time += 1
                else:
                    stab_time = 0
                
                error_pred = error

#TEST
    def test(self, X_test, y_test):
        Z1 = np.matmul(self.params['W1'], X_test) + self.params['b1']
        A1 = sigmoid(Z1)
        Z2 = np.matmul(self.params['W2'], A1) + self.params['b2']
        A2 = sigmoid(Z2)
        pred = np.argmax(A2, axis=0)

        accuracy = sum(pred == y_test) * 1 / len(y_test) * 100
        print('\tТочность: ' + str(round(accuracy, 2)) + '%')
