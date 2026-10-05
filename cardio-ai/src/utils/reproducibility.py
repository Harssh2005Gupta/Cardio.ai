"""
src/utils/reproducibility.py
------------------------------
Single place that seeds every RNG the project touches (Python's random,
numpy, TensorFlow). Call set_global_seed() once at the top of any training
or evaluation entrypoint.
"""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
