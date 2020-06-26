import tensorflow as tf


def train(model, data, checkpoint_path):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
    model.fit(data, epochs=5, callbacks=[checkpoint])
