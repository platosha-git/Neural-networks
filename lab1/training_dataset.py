import gzip
import numpy as np
from sklearn.preprocessing import OneHotEncoder

SIZE_OF_ONE_IMAGE = 28 ** 2

NUMBER_COUNT = 10

def read_dataset():
    with gzip.open('train-labels-idx1-ubyte.gz') as train_labels:
        data_from_train_file = train_labels.read()

    # Пропускаем первые 8 байт (метаданные)
    label_data = data_from_train_file[8:]
    assert len(label_data) == 60000

    # Конвертируем каждый байт в целое число.
    # Это будет число от 0 до 9
    labels = [int(label_byte) for label_byte in label_data]
    assert min(labels) == 0 and max(labels) == 9
    assert len(labels) == 60000

    images = []

    # Перебор тренировочного файла и чтение одного изображения за раз
    with gzip.open('train-images-idx3-ubyte.gz') as train_images:
        train_images.read(4 * 4)
        for _ in range(60000):
            image = train_images.read(size=SIZE_OF_ONE_IMAGE)
            assert len(image) == SIZE_OF_ONE_IMAGE

            # Конвертировать в NumPy
            image_np = np.frombuffer(image, dtype='uint8') / 255
            images.append(image_np)

    images = np.array(images)

    labels_np = np.array(labels).reshape((-1, 1))

    encoder = OneHotEncoder(categories='auto')
    labels_np_onehot = encoder.fit_transform(labels_np).toarray()
    return images, labels_np_onehot
