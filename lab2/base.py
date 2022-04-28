import numpy as np

M = 10000
A_SIZE = 100
R_SIZE = 10

def define_sigParams(s_size, a_size, r_size):
    return {'W1': np.random.randn(a_size, s_size) * np.sqrt(1. / s_size),
            'b1': np.zeros((a_size, 1)),
            'W2': np.random.randn(r_size, a_size) * np.sqrt(1. / a_size),
            'b2': np.zeros((r_size, 1))
        }

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def dSigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)

def dTanh(z):
    return 1 / (np.cosh(z)**2)


def ReLU(z):
    return np.maximum(0,z)

def dReLU(z):
    return (z > 0) * 1

