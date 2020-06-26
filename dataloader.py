import tensorflow as tf


def load_data(path):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    return generator.flow_from_directory(directory=path, target_size=(331, 331))

