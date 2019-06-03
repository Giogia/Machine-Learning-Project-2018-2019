import tensorflow as tf
from tensorflow import keras

IMG_X, IMG_Y = 28, 28


class CNN:

    """
    Usage:

    feature_extractor = CNN()
    training_features, evaluation_features, test_features = feature_extractor.extract(training_data, evaluation_data, test_data)

    """

    def __init__(self):

        input_shape = (IMG_X, IMG_Y, 1)

        self.model = keras.Sequential([
                        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
                        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
                        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                        keras.layers.Dropout(0.6),

                        keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu'),
                        keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
                        keras.layers.Dropout(0.6),

                        keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
                        keras.layers.Conv2D(512, kernel_size=(4, 4), activation='relu'),
                        keras.layers.BatchNormalization(),
                        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
                        keras.layers.Dropout(0.8),
                        keras.layers.Flatten()
                        ])

        self.model.load_weights('pretrained_model.h5')

    def extract(self, *data):

        sets = []

        for element in data:

            # Reshape every set
            element = element.reshape(element.shape[0], IMG_X, IMG_Y, 1)

            # Append high level features to results array
            sets.append(self.model.predict(element))

        return sets

