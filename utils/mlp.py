from jax import numpy as np
from jax.nn import sigmoid
from jax.random import normal, split, PRNGKey

# typing
from jax import Array
from typing import Tuple, Sequence, NamedTuple


def init_layer_params(m: int, n: int, *, key: Array) -> Tuple[Array, Array]:
    return normal(key, (n, m)) / np.sqrt(m), np.zeros(n)


def init_mlp_params(sizes: Sequence[int], *, key: Array) -> Sequence[Tuple[Array, Array]]:
    keys = split(key, len(sizes))
    return [init_layer_params(m, n, key=k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


class MLP(NamedTuple):
    params: Sequence[Tuple[Array, Array]]

    def __call__(self, x: Array) -> Array:
        for w, b in self.params[:-1]:
            x = sigmoid(np.dot(w, x) + b)
        w, b = self.params[-1]
        return sigmoid(np.dot(w, x) + b)


if __name__ == '__main__':
    layer_sizes = [784, 512, 512, 10]
    params = init_mlp_params(layer_sizes, key=PRNGKey(0))
    mlp = MLP(params)
    x = np.ones(784)
    print(mlp(x))
