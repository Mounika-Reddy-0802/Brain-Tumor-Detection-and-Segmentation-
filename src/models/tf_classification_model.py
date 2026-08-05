from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, layers

from src.models.backbone import build_efficientnet_b3


def build_classification_model(
    num_classes: int = 4,
    img_size: int = 224,
    dropout: float = 0.4,
) -> Model:
    """4-class classifier: glioma, meningioma, notumor, pituitary.

    Expects inputs in [0, 255]; EfficientNet rescales and normalizes internally.
    """
    base = build_efficientnet_b3(img_size=img_size, pretrained=True, pooling="avg")
    base.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout / 2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    return Model(inputs, outputs, name="classification_model")
