import struct
import numpy as np
import os

path = os.path.join(os.path.expanduser('./'), 'MNIST')

class RumelHurtNetwork():
    def __init__(self, n_x):
        self.m = 10000  # batch size  10000
        self.n_x = n_x # размерность входного слоя (784)
        self.n_h = 100  # размерность скрытого слоя 100
        self.eta =1  # скорость обучения
        self.epoch = 300  # количество эпох 300

        self.params = {'W1': np.random.randn(self.n_h, self.n_x) * np.sqrt(1. / self.n_x),  # веса от входного слоя к скрытому (матрица)
                      'b1': np.zeros((self.n_h, 1)),
                      'W2': np.random.randn(10, self.n_h) * np.sqrt(1. / self.n_h),
                      'b2': np.zeros((10, 1))
                  }

    def train(self, X_train, y_train, X_test, y_test):
        stab_time = 0
        error_pred = 10**9
        i = 0
        while stab_time < 10 and i < self.epoch:
            # перемешивание паттернов на каждой эпохе, чтоб не было зависимости обучения от порядка паттернов
            idx = np.random.permutation(X_train.shape[1])[:self.m]
            X = X_train[:, idx]  # известные входы
            y = y_train[:, idx]  # известные выходы для входов выше   784 yf 10.000
            # полуаем ответ от нейронки - прямой ход нейросети и сохраняет промежуточ знач-я нейросети
            forwardPass = self.forward(X)
            # обратное распространение ошибки - получаем коэфф для изменения
            gradient = self.back(X, y, forwardPass, i)
            # updating weights
            self.params = self.updater(gradient)
            if i % 10 == 0:
                print(f"Завершена эпоха {i}")
            i += 1

            # проверяем результат на тест выборке
            if i > 100:
                y_net_test = self.forward(X_test)['A2']
                error = np.sum(np.abs(y_net_test - y_test))
                if error >= error_pred:
                    stab_time += 1
                else:
                    stab_time = 0
                error_pred = error


        print(f"Завершена эпоха {self.epoch}")

        # for i in range(self.epoch):
        #     # перемешивание паттернов на каждой эпохе, чтоб не было зависимости обучения от порядка паттернов
        #     idx = np.random.permutation(X_train.shape[1])[:self.m]
        #     X = X_train[:, idx]  # известные входы
        #     y = y_train[:, idx]  # известные выходы для входов выше   784 yf 10.000
        #     # print('X:', len(X), len(X[0]))
        #     # полуаем ответ от нейронки - прямой ход нейросети и сохраняет промежуточ знач-я нейросети
        #     forwardPass = self.forward(X)
        #     # обратное распространение ошибки - получаем коэфф для изменения
        #     gradient = self.back(X, y, forwardPass)
        #     # updating weights
        #     self.params = self.updater(gradient)
        #     if i % 10 == 0:
        #         print(f"Завершена эпоха {i}")
        # print(f"Завершена эпоха {self.epoch}")

    def test(self, X_test):
        Z1 = np.matmul(self.params['W1'], X_test) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.params['W2'], A1) + self.params['b2']
        A2 = self.sigmoid(Z2) # выход нйросети
        pred = np.argmax(A2, axis=0)
        return pred

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    # производная сигмоиды
    def dSigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    # прямой ход вычислений
    def forward(self, X):
        forwardPass = {}
        forwardPass['Z1'] = np.matmul(self.params['W1'], X) + self.params['b1']
        forwardPass['A1'] = self.sigmoid(forwardPass['Z1'])
        forwardPass['Z2'] = np.matmul(self.params['W2'], forwardPass['A1']) + self.params['b2']
        forwardPass['A2'] = self.sigmoid(forwardPass['Z2'])     
        return forwardPass

    # обратное распространение ошибки
    def back(self, X, y, forwardPass, epoch):
        m = X.shape[1] # размер паттернов
        gradient = {}
       # 1/m - из-за пакетной обработки(прогонка эпохи) -  усреднение
        # 4. l - ошибка выходного слоя
        gradient['dZ2'] = (forwardPass['A2'] - y) # ошибка
        

        # 5. Z2 - суммирование без активации
        # 3.вычисление  локального градиента
        delta = gradient['dZ2'] * self.dSigmoid(forwardPass['Z2'])
        # 2. dW - высчитывание изменения веса без скорости обучения
        gradient['dW2'] =  (1. / m) * np.matmul(delta, forwardPass['A1'].T) # A1 - выход предыдущего слоя(у(l-1))
        gradient['db2'] =  (1. / m) * np.sum(delta, axis=1, keepdims=True) 
        



        # 4. - вычисление ошибки( сумма локального градиента на веса следующего слоя)  
        gradient['dA1'] =  np.matmul(self.params['W2'].T, delta)
        
        if (epoch == 299):
            print('Ошибка: ', gradient['dA1'])
        # 3. локальный градиент
        gradient['dZ1'] = gradient['dA1'] * self.dSigmoid(forwardPass['Z1'])
        # 2.
        gradient['dW1'] =(1. / m) * np.matmul(gradient['dZ1'], X.T)
        gradient['db1'] =(1. / m) * np.sum(gradient['dZ1'])

        return gradient

    def updater(self, grad):
        updatedParams = {}
        updatedParams['W2'] = self.params['W2'] - self.eta * grad['dW2']
        updatedParams['b2'] = self.params['b2'] - self.eta * grad['db2']
        updatedParams['W1'] = self.params['W1'] - self.eta * grad['dW1']
        updatedParams['b1'] = self.params['b1'] - self.eta * grad['db1']
        return updatedParams


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def oneHotEncoding(label):
    n = np.max(label) + 1
    v = np.eye(n)[label]
    return v.T


def imageProcess(data):
    data = data / 255
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    return data.T

if __name__ == '__main__':
    X_train = imageProcess(read_idx(path + '/train-images.idx3-ubyte'))
    y_train = oneHotEncoding(read_idx(path + '/train-labels-idx1-ubyte'))
    X_test = imageProcess(read_idx(path + '/t10k-images-idx3-ubyte'))
    y_test = read_idx(path + '/t10k-labels-idx1-ubyte')

    np.random.seed(7)
    network = RumelHurtNetwork(X_train.shape[0])
    # print(X_train.shape[0])
    network.train(X_train, y_train, X_test, y_test)
    y_answer = network.test(X_test)

    print('Accuracy:', sum(y_answer == y_test) * 1 / len(y_test))

