import struct
import numpy as np
import gzip
from sklearn.preprocessing import OneHotEncoder


LABELS_FILE_TRAIN = './MNIST/train-labels-idx1-ubyte'
LABELS_FILE_TEST = './MNIST/t10k-labels-idx1-ubyte'

IMAGES_FILE_TRAIN = './MNIST/train-images.idx3-ubyte'
IMAGES_FILE_TEST = './MNIST/t10k-images-idx3-ubyte'


def get_labels(filename):
	with open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), 'uint8').reshape(shape)


def get_images(filename):
	data = get_labels(filename)
	data = data / 255
	data = data.reshape(data.shape[0],data.shape[1]*data.shape[2])

	return data.T


def get_dataset():
	X_train = get_images(IMAGES_FILE_TRAIN)
	X_test = get_images(IMAGES_FILE_TEST)

	y_train = get_labels(LABELS_FILE_TRAIN)
	n = np.max(y_train) + 1
	v = np.eye(n)[y_train]
	y_train = v.T

	y_test = get_labels(LABELS_FILE_TEST)

	return X_train, X_test, y_train, y_test
