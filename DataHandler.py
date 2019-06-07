from tensorflow import keras
import numpy as np
from Sets import Sets
from sklearn.preprocessing import StandardScaler

STD_SCALER = 'std'


def randomize(a, b):

    # Generate the permutation index array.
    s = np.arange(a.shape[0])
    np.random.shuffle(s)

    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[s]
    shuffled_b = b[s]

    return shuffled_a, shuffled_b


def load_data(eval_percentage=0.2, scaler_kind=None):
    fashion_mnist = keras.datasets.fashion_mnist

    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    train_x = train_x.reshape((-1, train_x.shape[1]*train_x.shape[2]))
    test_x = test_x.reshape((-1, test_x.shape[1]*test_x.shape[2]))

    train_x, train_y = randomize(train_x, train_y)

    train_x = train_x.astype(np.float)
    test_x = test_x.astype(np.float)

    if scaler_kind is None:
        train_x = train_x / 255.0
        test_x = test_x / 255.0
    elif scaler_kind == STD_SCALER:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.fit_transform(test_x)

    limit = int(len(train_x) * eval_percentage)
    sets = Sets(train_x[limit:], train_y[limit:], train_x[:limit], train_y[:limit], test_x, test_y)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return sets, class_names


def reshuffle(sets, eval_percentage=0.2):

    train_x = np.append(sets.train.x, sets.eval.x, axis=0)
    train_y = np.append(sets.train.y, sets.eval.y, axis=0)

    train_x, train_y = randomize(train_x, train_y)

    limit = int(len(train_x) * eval_percentage)
    sets = Sets(train_x[limit:], train_y[limit:], train_x[:limit], train_y[:limit], sets.test.x, sets.test.y)

    return sets
