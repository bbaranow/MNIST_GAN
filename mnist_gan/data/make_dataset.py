from tensorflow.keras.datasets.fashion_mnist import load_data
import tensorflow as tf

def download_dataset():
    '''
    Function to download dataset to memory
    '''
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    return train_images, train_labels

def preprocess_dataset(train_images):
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    return train_images

def make_tf_dataset(train_images, BUFFER_SIZE = 60000, BATCH_SIZE = 256):
    train_dataset = tf.data.Dataset\
                        .from_tensor_slices(train_images)\
                        .shuffle(BUFFER_SIZE)\
                        .batch(BATCH_SIZE)
    return train_dataset