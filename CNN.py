import tensorflow as tf
from tensorflow import keras

# TODO Ancora tutto, è una prova per vedere se si riusciva a instanziare solo la parte di feature extraction ed eliminare i dense layers
class CNN:

    def __init__(self):

        img_x, img_y = 28, 28
        input_shape = (img_x, img_y, 1)

        self.model = keras.Sequential([
                        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=input_shape),
                        keras.layers.Activation(tf.nn.relu),
                        keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                        keras.layers.Conv2D(64, (5, 5)),
                        keras.layers.Activation(tf.nn.relu),
                        keras.layers.SpatialDropout2D(0.5),
                        keras.layers.MaxPooling2D(pool_size=(2, 2)),
                        keras.layers.Flatten()
                        ])

        self.model.load_weights('pretrained_model.h5')

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


feature_extractor = CNN()
