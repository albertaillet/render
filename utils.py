from jax import numpy as np
from jax.nn import logsumexp

# typing
from jax import Array


def norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Array:
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x: Array, axis: int = -1, eps: float = 1e-20) -> Array:
    return x / norm(x, axis=axis, keepdims=True, eps=eps)


def min(x: Array, axis: int = -1, keepdims: bool = False) -> Array:
    return np.min(x, axis=axis, keepdims=keepdims)


def softmin(x: Array, c: float = 8.0, axis: int = -1, keepdims: bool = False) -> Array:
    return -logsumexp(-c * x, axis=axis, keepdims=keepdims) / c