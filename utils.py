from jax import numpy as np

# typing
from jax import Array


def norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Array:
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x: Array, axis: int = -1, eps: float = 1e-20) -> Array:
    return x / norm(x, axis=axis, keepdims=True, eps=eps)
