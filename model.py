import tensorflow as tf


def ProductDetectionModel():
    base_model = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg")
    base_model.trainable = False
    #
    predict_model = tf.keras.Sequential([base_model, tf.keras.layers.Dense(42)])
    predict_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return predict_model
