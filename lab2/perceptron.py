import numpy as np

M = 10000
ETA = 1
LAMBA = 2

def softMax(X):
    e = np.exp(X)
    p = e/np.sum(e, axis=0)
    return p

def ReLU(z):
    return np.maximum(0,z)


def sigmoid(z):
    return 1./(1.+np.exp(-z))


def tanh(z):
    return np.tanh(z)

def dReLU(z):
    return (z > 0) * 1

def dSigmoid(z):
    return sigmoid(z) *(1-sigmoid (z))

def dTanh(z):
    return 1/(np.cosh(z)**2)

def crossEntropyR2(y, y_hat, lamda, params):
    m = y.shape[1]
    cost = -(1/m) * np.sum(y*np.log(y_hat)) + lamda/(2*m) * (np.sum(params['W1']**2) + np.sum(params['W2']**2))
    return cost

class Perceptron():
    def __init__(self, params, f_activation, b_activation):
        self.params = params
        self.f_activation = f_activation
        self.b_activation = b_activation

    def forward(self, X):
        forwardPass = {}
        forwardPass['Z1'] = np.matmul(self.params['W1'], X) + self.params['b1']
        forwardPass['A1'] = self.f_activation(forwardPass['Z1'])
        forwardPass['Z2'] = np.matmul(self.params['W2'],forwardPass['A1']) + self.params['b2']
        forwardPass['A2'] = softMax(forwardPass['Z2'])
        return forwardPass

    def back(self, X, y, forwardPass):
        m = X.shape[1]
        gradient = {}
        gradient['dZ2'] = forwardPass['A2'] - y
        gradient['dW2'] = (1./m) * np.matmul(gradient['dZ2'], forwardPass['A1'].T)
        gradient['db2'] = (1./m) * np.sum(gradient['dZ2'], axis=1, keepdims=True)
        gradient['dA1'] = np.matmul(self.params['W2'].T, gradient['dZ2'])
        gradient['dZ1'] = gradient['dA1'] * self.b_activation(forwardPass['Z1'])
        gradient['dW1'] = (1./m) * np.matmul(gradient['dZ1'], X.T)
        gradient['db1'] = (1./m) * np.sum(gradient['dZ1'])
        
        return gradient

    def updater(self, grad):
        self.params['W2'] -= ETA * grad['dW2']
        self.params['b2'] -= ETA * grad['db2']
        self.params['W1'] -= ETA * grad['dW1']
        self.params['b1'] -= ETA * grad['db1']

    def train(self, X_train, y_train, epoch):
        for i in range(epoch):
            idx = np.random.permutation(X_train.shape[1])[:M]
            X = X_train[:,idx]
            y = y_train[:,idx]

            forwardPass = self.forward(X)
            gradient = self.back(X, y, forwardPass)
            self.updater(gradient)
            
            if i % 10 == 0:
                print('\tЭпоха ' + str(i))

    def test(self, X_test, y_test):
        Z1 = np.matmul(self.params['W1'], X_test) + self.params['b1']
        A1 = self.f_activation(Z1)
        Z2 = np.matmul(self.params['W2'],A1) + self.params['b2']
        A2 = softMax(Z2)
        pred = np.argmax(A2, axis=0)
        print('Accuracy:',sum(pred==y_test)*1/len(y_test))
