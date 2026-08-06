"""Shared EfficientNetB3 backbone construction.

Keras 3.11 cannot load the ImageNet weights for the V1 EfficientNets: the
builder inserts an unadapted Normalization layer, which shifts the index-based
.h5 weight loader by one and raises

    ValueError: Shape mismatch in layer #1 (named stem_conv) ...

EfficientNetV2 and ResNet are unaffected, so this is specific to the family this
project uses. Loading the same file by layer name sidesteps the index shift and
restores every convolution and batch-norm weight, so build_efficientnet_b3
tries the normal path first and falls back to name-based loading.

Keras 3.13 fixes this, and requirements.txt pins that, so the fallback should
never run in a correctly-installed environment. It is kept as a safety net.

Be aware the fallback is not perfectly equivalent: Keras only inserts the second
(ImageNet stddev) Rescaling layer on the weights="imagenet" path, so a model
built here through the fallback has a two-layer stem preamble where a natively
built one has three. That is harmless for training from scratch, but it means
weights saved by a newer Keras will not align if loaded into a fallback-built
model. Upgrade Keras rather than relying on this path.
"""

from __future__ import annotations

import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB3

WEIGHTS_URL = "https://storage.googleapis.com/keras-applications/efficientnetb3_notop.h5"
WEIGHTS_FILE = "efficientnetb3_notop.h5"


def build_efficientnet_b3(img_size: int = 224, pretrained: bool = True, pooling: str = "avg") -> Model:
    """EfficientNetB3 feature extractor with ImageNet weights.

    The returned model expects inputs in [0, 255]: it carries its own
    Rescaling(1/255) and Normalization layers.
    """
    kwargs = {
        "include_top": False,
        "input_shape": (img_size, img_size, 3),
        "pooling": pooling,
    }

    if not pretrained:
        return EfficientNetB3(weights=None, **kwargs)

    try:
        return EfficientNetB3(weights="imagenet", **kwargs)
    except ValueError as exc:
        if "stem_conv" not in str(exc):
            raise

    base = EfficientNetB3(weights=None, **kwargs)
    weights_path = keras.utils.get_file(WEIGHTS_FILE, WEIGHTS_URL, cache_subdir="models")
    base.load_weights(weights_path, by_name=True, skip_mismatch=True)
    print(f"Loaded ImageNet weights by layer name (Keras {keras.__version__} index-loader workaround)")
    return base
