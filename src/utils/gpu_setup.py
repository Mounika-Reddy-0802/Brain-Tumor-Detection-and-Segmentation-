from __future__ import annotations

import tensorflow as tf
import torch


def configure_gpu(verbose: bool = True) -> None:
    """Enable TensorFlow memory growth so TensorFlow and PyTorch can share GPU."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if verbose:
        print(f"TF GPUs available: {len(gpus)}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
