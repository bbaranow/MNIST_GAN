#Function for Generator of GAN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_generator_model():
    
    seed = tf.keras.Input(shape=((100,)))

    x = layers.Dense(7*7*256, use_bias=False)(seed)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)
    
    x = layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(1, (5,5), strides=(1,1), padding='same', activation='tanh', use_bias=False)(x)
    
    model = tf.keras.Model(inputs=seed, outputs=x)

    return model

def make_discriminator_model():
    image = tf.keras.Input(shape=((28,28,1)))
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding = 'same', use_bias=False)(image)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding = 'same', use_bias=False)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    model = tf.keras.Model(inputs = image, outputs = x)
    return model