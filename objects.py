from jax import numpy as np
from jax import tree_map
from utils.linalg import norm, softmax
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


class Camera(NamedTuple):
    up: Vector
    position: Vector
    target: Vector


class Scene(NamedTuple):
    objects: Tuple[Objects, ...]
    camera: Camera

    def sdf(self, p: Array) -> Array:
        return np.concatenate([o.sdf(p) for o in self.objects])

    def color(self, p: Array) -> Array:
        dists = self.sdf(p)
        colors = np.concatenate([o.color for o in self.objects])
        return softmax(-8.0 * dists) @ colors


def get_scene(scene_dict: dict) -> Tuple[Scene, Tuple[int, int]]:
    '''Create a scene from a dict of the form:
    {
        'width': 800
        'height': 600
        'Camera': {
            'position': [0, 0, 5]
            'target': [0, 0, 0]
            'up': [0, 1, 0]
        }
        'Objects': [
            {'Sphere': {'position': [0, 0, 0], 'radius': 0.5, 'color': [0, 0, 1]}},
            {'Plane': {'position': [0, 0, 0], 'normal': [0, 1, 0], 'color': [1, 1, 1]}},
        ]
    }
    '''

    def is_leaf(node: Any) -> bool:
        return isinstance(node, (list, int))

    view_size = scene_dict['width'], scene_dict['height']
    camera_dict = scene_dict['Camera']
    objects_dict = scene_dict['Objects']

    camera = Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    object_dicts = defaultdict(list)
    for outer_obj_dict in objects_dict:
        for obj_type, obj in outer_obj_dict.items():  # there should only be one key
            object_dicts[obj_type].append(obj)

    for obj_type, objs in object_dicts.items():
        while len(objs) % 5 != 0:
            # pad with objects to multiples of 5:
            if obj_type == 'Sphere':
                objs.append(
                    {'position': [0, -1_000, 0], 'radius': 0, 'color': [1, 0, 0]}
                )
            elif obj_type == 'Plane':
                objs.append(
                    {
                        'position': [0, -1_000, 0],
                        'normal': [0, 1, 0],
                        'color': [1, 0, 0],
                    }
                )

    objects = []
    for obj_type, objs in object_dicts.items():
        transposed_objs = tree_map(lambda *xs: list(xs), *objs, is_leaf=is_leaf)
        kwargs = tree_map(np.float32, transposed_objs, is_leaf=is_leaf)
        objects.append(globals()[obj_type + 's'](**kwargs))

    return Scene(objects=tuple(objects), camera=camera), view_size


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene from a dict of the form:
    {
        'width': 800
        'height': 600
        'Camera': {
            'position': [0, 0, 5]
            'target': [0, 0, 0]
            'up': [0, 1, 0]
        }
        'Objects': [
            {'Sphere': {'position': [0, 0, 0], 'radius': 0.5, 'color': [0, 0, 1]}},
            {'Plane': {'position': [0, 0, 0], 'normal': [0, 1, 0], 'color': [1, 1, 1]}},
        ]
    }
    '''
    check_dict_fields(scene_dict.get('Camera'), Camera)

    for outer_obj_dict in scene_dict.get('Objects'):
        for obj_type, obj in outer_obj_dict.items():
            check_dict_fields(obj, globals()[obj_type + 's'])


def check_dict_fields(obj: dict, cls: type) -> None:
    type_hints = get_type_hints(cls)

    if not isinstance(obj, dict):
        raise TypeError(f'{obj} must be a dict, not {type(obj)}')

    provided_fields = set(obj.keys())
    required_fields = set(type_hints.keys())
    if provided_fields != required_fields:
        raise ValueError(
            f'{obj} must have fields {required_fields}, not {provided_fields}'
        )

    for field in obj:
        if not isinstance(obj[field], type_hints[field]):
            raise TypeError(
                f'Field {field} must be of type {type_hints[field]}, not {type(obj[field])}'
            )
        if type_hints[field] == Vector:
            if len(obj[field]) != 3:
                raise ValueError(f'Field {field} must be a 3-vector, not {obj[field]}')
            for x in obj[field]:
                if not isinstance(x, (int, float)):
                    raise TypeError(
                        f'Field {field} of object must be a 3-vector of floats, not {obj[field]}'
                    )


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)
    check_scene_dict(scene_dict)
    scene = get_scene(scene_dict)
