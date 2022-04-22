import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder

SIZE_OF_ONE_IMAGE = 28 ** 2
NUM_OF_COUNTS = 10
NUM_OF_IMAGES = 60000

L_BYTES_OF_METADATA = 8
LABELS_FILE = 'train-labels-idx1-ubyte.gz'

I_BYTES_OF_METADATA = 16
IMAGES_FILE = 'train-images-idx3-ubyte.gz'


def get_labels():
    with gzip.open(LABELS_FILE) as labels_f:
        labels_f.read(L_BYTES_OF_METADATA)
        labels_bytes = labels_f.read()

    labels_int = []
    for label in labels_bytes:
        labels_int.append(int(label))

    return labels_int


def get_images():
    images = []

    with gzip.open(IMAGES_FILE) as images_f:
        images_f.read(I_BYTES_OF_METADATA)

        for i in range(NUM_OF_IMAGES):
            image = images_f.read(SIZE_OF_ONE_IMAGE)
            image_np = np.frombuffer(image, 'uint8') / 255
            images.append(image_np)

    return images

def get_dataset():
    labels = get_labels()
    labels_np = np.array(labels).reshape((-1, 1))
    encoder = OneHotEncoder(categories='auto')
    labels_array = encoder.fit_transform(labels_np).toarray()

    images = get_images()
    images = np.array(images)

    return labels_array, images
