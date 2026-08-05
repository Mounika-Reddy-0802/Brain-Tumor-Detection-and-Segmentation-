from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.models.backbone import build_efficientnet_b3


def build_detection_model(img_size: int = 224, dropout: float = 0.4) -> Model:
    """Binary classifier: tumor (1) vs no-tumor (0).

    Expects inputs in [0, 255]; EfficientNet rescales and normalizes internally.
    """
    base = build_efficientnet_b3(img_size=img_size, pretrained=True, pooling="avg")
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    return Model(inputs, outputs, name="detection_model")
