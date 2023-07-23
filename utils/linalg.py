from jax import nn, numpy as np, Array
from typing import Union


def norm(x: Array, axis: int = -1, keepdims: bool = False, eps: float = 0.0) -> Array:
    return np.sqrt(np.square(x).sum(axis, keepdims=keepdims).clip(eps))


def normalize(x: Array, axis: int = -1, eps: float = 1e-20) -> Array:
    return x / norm(x, axis=axis, keepdims=True, eps=eps)


def smoothmin(x: Array, c: Union[float, Array] = 0.125, **kwargs) -> Array:
    return -c * nn.logsumexp(-x / c, **kwargs)


def smoothabs(x: Array, s: float = 1e-3) -> Array:
    return np.sqrt(np.square(x) + s)


def softmax(x: Array, *args, **kwargs) -> Array:
    return nn.softmax(x, *args, **kwargs)


def relu(x: Array, *args, **kwargs) -> Array:
    return nn.relu(x, *args, **kwargs)


def Rx(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(theta: float) -> Array:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Rxyz(rotation_angles: Array) -> Array:
    theta, phi, psi = rotation_angles
    return Rz(psi) @ Ry(phi) @ Rx(theta)
