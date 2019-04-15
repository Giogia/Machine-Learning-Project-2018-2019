from tensorflow import keras
import numpy as np
from Sets import Sets


def randomize(a, b):

    # Generate the permutation index array.
    s = np.arange(a.shape[0])
    np.random.shuffle(s)

    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[s]
    shuffled_b = b[s]

    return shuffled_a, shuffled_b


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    train_x, train_y = randomize(train_x, train_y)
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    limit = int(len(train_x)*0.2)
    sets = Sets(train_x[limit:], train_y[limit:], train_x[:limit], train_y[:limit], test_x, test_y)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return sets, class_names

def load_linearized_data():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    train_x = train_x.reshape((-1,784))
    test_x = test_x.reshape((-1,784))

    train_x, train_y = randomize(train_x, train_y)
    train_x = train_x / 255.0
    test_x = test_x / 255.0

    limit = int(len(train_x)*0.2)
    sets = Sets(train_x[limit:], train_y[limit:], train_x[:limit], train_y[:limit], test_x, test_y)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return sets, class_names
