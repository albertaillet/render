from jax import numpy as np
from jax import tree_map
from utils import norm, softmax
from collections import defaultdict

# typing
from jax import Array
from typing import NamedTuple, Sequence, Any


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


def get_scene(scene_dict: dict) -> Scene:
    '''Create a scene from a dict of the form:
    scene_dict = [
        {'Sphere': {'position': [0, 0, 0], 'radius': 0.5, 'color': [0, 0, 1]}},
        {'Plane': {'position': [0, 0, 0], 'normal': [0, 1, 0], 'color': [1, 1, 1]}},
    ]
    '''

    def is_leaf(node: Any) -> bool:
        return isinstance(node, list)

    object_dicts = defaultdict(list)
    for outer_obj_dict in scene_dict:
        for obj_type, obj in outer_obj_dict.items():  # there should only be one key
            object_dicts[obj_type].append(obj)

    objects = []
    for obj_type, objs in object_dicts.items():
        transposed_objs = tree_map(lambda *xs: list(xs), *objs, is_leaf=is_leaf)
        kwargs = tree_map(np.array, transposed_objs, is_leaf=is_leaf)
        objects.append(globals()[obj_type + 's'](**kwargs))

    return Scene(objects=tuple(objects))


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene from a dict of the form:
    scene_dict = [
        {'Sphere': {'position': [0, 0, 0], 'radius': 0.5, 'color': [0, 0, 1]}},
        {'Plane': {'position': [0, 0, 0], 'normal': [0, 1, 0], 'color': [1, 1, 1]}},
    ]
    '''
    for i, outer_obj_dict in enumerate(scene_dict, 1):
        for obj_type, obj in outer_obj_dict.items():
            if obj_type + 's' not in globals():
                raise ValueError(f'Unknown object type: {obj_type}')
            obj_class = globals()[obj_type + 's']
            if not isinstance(obj, dict):
                raise ValueError(f'Object must be a dict, not {type(obj)}')

            provided_fields = set(obj.keys())
            required_fields = set(obj_class._fields)
            if provided_fields != required_fields:
                raise ValueError(
                    f'Object {i} ({obj_type}) must have fields {required_fields}, not {provided_fields}'
                )

            for k, v in obj.items():
                if isinstance(v, Sequence):
                    if len(v) != 3:
                        raise ValueError(
                            f'Object {i} ({obj_type}):\n Vector {k}:{v} must have 3 elements, not {len(v)}'
                        )
                    for x in v:
                        if not isinstance(x, (float, int)):
                            raise TypeError(
                                f'Object {i} ({obj_type}):\n Vector {k}:{v} must only contain floats, not {type(x)}'
                            )

                if not isinstance(v, Sequence) and not isinstance(v, (int, float)):
                    raise ValueError(
                        f'Value must be a float or a vector, not {type(v)}'
                    )


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scene.yml', 'r'), SafeLoader)
    scene = get_scene(scene_dict)
