import tensorflow as tf
from tensorflow import keras

class DenoiseAutoEncoder(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.encoder_ = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),
            keras.layers.Conv2D(16, 3, padding='same', strides=2, activation='relu'),
            keras.layers.Conv2D(8, 3, padding='same', strides=2, activation='relu')
        ])
        self.decoder_ = keras.Sequential([
            keras.layers.Conv2DTranspose(8, 3, padding='same', strides=2, activation='relu'),
            keras.layers.Conv2DTranspose(16, 3, padding='same', strides=2, activation='relu'),
            keras.layers.Conv2D(1, 3, padding='same', strides=1, activation='sigmoid')
        ])

    def call(self, x):
        latent_space = self.encoder_(x)
        output_ = self.decoder_(latent_space)
        return output_
