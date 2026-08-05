from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_image
from src.utils.project import EFFICIENTNET_INPUT_SCALE


def tf_load_raw(image_path: tf.Tensor, label: tf.Tensor, img_size: int = 224) -> tuple[tf.Tensor, tf.Tensor]:
    """Load an unprocessed MRI through the OpenCV pipeline (crop, resize, CLAHE).

    Uses tf.py_function, so it runs in Python and does not parallelise well.
    Prefer tf_load_preprocessed when the images have already been written to
    data/processed by src.data.preprocessing.
    """

    def _load(path: tf.Tensor) -> np.ndarray:
        decoded_path = path.numpy().decode("utf-8")
        return preprocess_image(decoded_path, img_size).astype(np.float32)

    img = tf.py_function(func=_load, inp=[image_path], Tout=tf.float32)
    img.set_shape([img_size, img_size, 3])
    return img, label


def tf_load_preprocessed(image_path: tf.Tensor, label: tf.Tensor, img_size: int = 224) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode an already-preprocessed image using native TensorFlow ops.

    The crop/CLAHE work is already baked into the files on disk, so this only
    decodes and resizes. Re-running preprocess_image here would apply CLAHE a
    second time and desync training from the inference pipeline.
    """
    raw = tf.io.read_file(image_path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method=tf.image.ResizeMethod.AREA)
    return img / 255.0, label


def apply_tf_augmentations(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """TensorFlow-native augmentations for the training dataset only.

    Expects and returns images in [0, 1]; the brightness and contrast deltas are
    calibrated for that range.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return tf.clip_by_value(img, 0.0, 1.0), label


def to_efficientnet_range(img: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale [0, 1] images up to the [0, 255] range EfficientNet expects."""
    return img * EFFICIENTNET_INPUT_SCALE, label


def build_tf_dataset(
    image_paths: list[str] | np.ndarray,
    labels: list[int] | np.ndarray,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = True,
    img_size: int = 224,
    preprocessed: bool = False,
) -> tf.data.Dataset:
    """Create an optimized tf.data input pipeline.

    Set preprocessed=True when image_paths point at data/processed, which swaps
    the Python/OpenCV loader for native TensorFlow decoding.
    """
    image_paths_np = np.array(image_paths)
    labels_np = np.array(labels)

    ds = tf.data.Dataset.from_tensor_slices((image_paths_np, labels_np))

    # Shuffle file paths rather than decoded images: a 1000-image buffer of
    # 224x224x3 float32 costs ~600 MB, while the same buffer of paths is free
    # and lets us reshuffle the whole split every epoch.
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths_np), reshuffle_each_iteration=True)

    loader = tf_load_preprocessed if preprocessed else tf_load_raw
    ds = ds.map(lambda p, l: loader(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(apply_tf_augmentations, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(to_efficientnet_range, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
