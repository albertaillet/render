from jax import numpy as np
from jax import tree_map
from utils import norm, softmax
from collections import defaultdict

# typing
from jax import Array
from typing import NamedTuple


Objects = NamedTuple


class Spheres(Objects):
    position: Array
    radius: Array
    color: Array

    def sdf(self, p: Array) -> Array:
        return norm(p - self.position) - self.radius


class Planes(Objects):
    position: Array
    normal: Array
    color: Array

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.position, self.normal)


class Scene(NamedTuple):
    objects: tuple[Objects, ...]

    def sdf(self, p: Array) -> Array:
        return np.concatenate([o.sdf(p) for o in self.objects])

    def color(self, p: Array) -> Array:
        dists = self.sdf(p)
        colors = np.concatenate([o.color for o in self.objects])
        return softmax(-8.0 * dists) @ colors


def get_scene(scene_json: dict) -> Scene:
    '''Create a scene from a JSON object of the form:
    [
        {
            'type': 'Sphere',
            'position': [0, 0, 0],
            'radius': 1,
            'color': [1, 0, 0]
        },
        {
            'type': 'Plane',
            'position': [0, 0, 0],
            'normal': [0, 1, 0],
            'color': [0, 1, 0]
        }
    ]
    '''

    def is_leaf(x):
        return isinstance(x, list)

    object_dicts = defaultdict(list)
    for obj in scene_json:
        obj_type = obj.pop('type')
        object_dicts[obj_type].append(obj)

    objects = []
    for obj_type, objs in object_dicts.items():
        transposed_objs = tree_map(lambda *xs: list(xs), *objs, is_leaf=is_leaf)
        kwargs = tree_map(np.array, transposed_objs, is_leaf=is_leaf)
        objects.append(globals()[obj_type + 's'](**kwargs))

    return Scene(objects=tuple(objects))


if __name__ == '__main__':
    import json
    scene_json = json.load(open('scene.json'))
    scene = get_scene(scene_json)