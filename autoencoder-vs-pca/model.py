import tensorflow as tf
from tensorflow import keras

def build_autoencoder(input_dim, latent_dim=10):
    encoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(latent_dim)
    ])

    decoder = keras.Sequential([
        keras.layers.InputLayer(input_shape=(latent_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(input_dim, activation='sigmoid')
    ])

    autoencoder = keras.Sequential([encoder, decoder])
    autoencoder.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return autoencoder, encoder
