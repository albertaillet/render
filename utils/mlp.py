from jax import numpy as np
from jax.nn import celu
from jax.random import normal, split, PRNGKey

# typing
from jax import Array
from typing import Tuple, Sequence
from jax.random import PRNGKeyArray


def init_layer_params(
    m: int, n: int, key: PRNGKeyArray, scale: float = 1e-2
) -> Tuple[Array, Array]:
    w_key, b_key = split(key)
    return scale * normal(w_key, (n, m)), scale * normal(b_key, (n,))


def init_mlp_params(
    sizes: Sequence[int], key: PRNGKeyArray
) -> Sequence[Tuple[Array, Array]]:
    keys = split(key, len(sizes))
    return [
        init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def forward_mlp(params: Sequence[Tuple[Array, Array]], x: Array) -> Array:
    activations = x
    for w, b in params[:-1]:
        outputs = np.dot(w, activations) + b
        activations = celu(outputs)

    w, b = params[-1]
    return np.dot(w, activations) + b


if __name__ == '__main__':
    layer_sizes = [784, 512, 512, 10]
    params = init_mlp_params(layer_sizes, PRNGKey(0))