from jax import numpy as np
from utils import norm

# typing
from jax import Array
from typing import NamedTuple


class Spheres(NamedTuple):
    pos: Array
    radii: Array

    def sdf(self, p: Array) -> Array:
        return norm(p - self.pos) - self.radii


class Planes(NamedTuple):
    pos: Array
    normal: Array

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.pos, self.normal)
