from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_image

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def tf_load_and_preprocess(image_path: tf.Tensor, label: tf.Tensor, img_size: int = 224) -> tuple[tf.Tensor, tf.Tensor]:
    """Wrap OpenCV preprocessing for tf.data using tf.py_function."""

    def _load(path: tf.Tensor) -> np.ndarray:
        decoded_path = path.numpy().decode("utf-8")
        img = preprocess_image(decoded_path, img_size)
        return img.astype(np.float32)

    img = tf.py_function(func=_load, inp=[image_path], Tout=tf.float32)
    img.set_shape([img_size, img_size, 3])
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, label


def apply_tf_augmentations(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow-native augmentations for training dataset only."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return img, label


def build_tf_dataset(
    image_paths: list[str] | np.ndarray,
    labels: list[int] | np.ndarray,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    img_size: int = 224,
) -> tf.data.Dataset:
    """Create optimized tf.data input pipeline with map, batch, and prefetch."""
    image_paths_np = np.array(image_paths)
    labels_np = np.array(labels)

    ds = tf.data.Dataset.from_tensor_slices((image_paths_np, labels_np))
    ds = ds.map(
        lambda p, l: tf_load_and_preprocess(p, l, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        ds = ds.map(apply_tf_augmentations, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(image_paths_np)), reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
