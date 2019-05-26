from tensorflow import keras
import tensorflow as tf
import numpy as np


class FCNN:

    def __init__(self, **kwargs):
        # Start neural network
        self.batch_size = kwargs['batch_size']
        self.epochs = kwargs['epochs']
        self.verbose = kwargs['verbose']
        self.optimizer = kwargs['optimizer']
        self.loss = kwargs['loss']
        self.metrics = kwargs['metrics']

        self.network = keras.models.Sequential([

            keras.layers.Dense(128),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.Dense(10),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(tf.nn.softmax)
        ])

        lr_dec = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, mode='auto',
                                                   min_delta=0.0001, cooldown=0, min_lr=0)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=1,
                                                   mode='auto', baseline=None, restore_best_weights=True)
        # Compile neural network
        self.network.compile(optimizer=self.optimizer,
                             loss=self.loss,
                             metrics=self.metrics,
                             callbacks=[lr_dec, early_stop])

    def fit(self, features, labels, eval_x, eval_label):
        self.network.fit(features,
                         labels,
                         batch_size=self.batch_size,
                         epochs=self.epochs,
                         validation_data=(eval_x, eval_label),
                         verbose=self.verbose)

    def predict(self, features):
        return np.argmax(self.network.predict(features), axis=1)

