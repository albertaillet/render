from jax import numpy as np
from jax import tree_map
from jax_utils import norm, softmax
from collections import defaultdict

# typing
from jax import Array
from typing import NamedTuple, Tuple, Sequence, Union, Any, get_type_hints

Objects = NamedTuple
Vector = Union[Array, Sequence]
Color = Union[Array, Sequence, str]
Scalar = Union[Array, float, int]


class Spheres(Objects):
    position: Vector
    radius: Scalar
    color: Color

    def sdf(self, p: Array) -> Array:
        return norm(p - self.position) - self.radius


class Planes(Objects):
    position: Vector
    normal: Vector
    color: Color

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.position, self.normal)


class Scene(NamedTuple):
    objects: Tuple[Objects, ...]

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
                raise TypeError(f'Object must be a dict, not {type(obj)}')

            type_hints = get_type_hints(obj_class)
            provided_fields = set(obj.keys())
            required_fields = set(type_hints.keys())
            if provided_fields != required_fields:
                raise ValueError(
                    f'Object {i} ({obj_type}) must have fields {required_fields}, not {provided_fields}'
                )

            for field in obj:
                if not isinstance(obj[field], type_hints[field]):
                    raise TypeError(
                        f'Field {field} of object {i} ({obj_type}) must be of type {type_hints[field]}, not {type(obj[field])}'
                    )
                if type_hints[field] == Vector:
                    if len(obj[field]) != 3:
                        raise ValueError(
                            f'Field {field} of object {i} ({obj_type}) must be a 3-vector, not {obj[field]}'
                        )
                    for x in obj[field]:
                        if not isinstance(x, (int, float)):
                            raise TypeError(
                                f'Field {field} of object {i} ({obj_type}) must be a 3-vector of floats, not {obj[field]}'
                            )


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)
    scene = get_scene(scene_dict)
