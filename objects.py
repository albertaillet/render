'''Objects for raymarching
Expected scene dict format:
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
from jax import numpy as np
from jax import tree_map
from utils.linalg import norm, softmax
from collections import defaultdict

# typing
from typeguard import check_type
from jaxtyping import Array, Float
from typing import NamedTuple, Tuple, Any, get_type_hints

Objects = NamedTuple
Vec3 = Float[Array, '3']
Vec3s = Float[Array, 'n 3']
Scalars = Float[Array, 'n']

UNFLATTENED_TYPES = {
    Vec3: Tuple[float, float, float],
    Vec3s: Tuple[float, float, float],
    Scalars: float,
}


class Spheres(Objects):
    position: Vec3s
    radius: Scalars
    color: Vec3s

    def sdf(self, p: Array) -> Array:
        return norm(p - self.position) - self.radius


class Planes(Objects):
    position: Vec3s
    normal: Vec3s
    color: Vec3s

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.position, self.normal)


class Camera(NamedTuple):
    up: Vec3
    position: Vec3
    target: Vec3


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
    '''Create a scene from a dict of expected format (see top of file)'''

    def is_leaf(node: Any) -> bool:
        return isinstance(node, list)

    view_size = scene_dict['width'], scene_dict['height']
    camera_dict = scene_dict['Camera']
    objects_dict = scene_dict['Objects']

    camera = Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    object_dicts = defaultdict(list)
    for outer_obj_dict in objects_dict:
        for obj_type, obj in outer_obj_dict.items():  # there should only be one key
            object_dicts[obj_type].append(obj)

    # pad with objects to multiples of 5 to avoid recompilation of jitted function
    for obj_type, objs in object_dicts.items():
        while len(objs) % 5 != 0:
            if obj_type == 'Sphere':
                objs.append({'position': [0, -100, 0], 'radius': 0, 'color': [1, 0, 0]})
            elif obj_type == 'Plane':
                objs.append(
                    {'position': [0, -100, 0], 'normal': [0, 1, 0], 'color': [1, 0, 0]}
                )

    objects = []
    for obj_type, objs in object_dicts.items():
        transposed_objs = tree_map(lambda *xs: list(xs), *objs, is_leaf=is_leaf)
        kwargs = tree_map(np.float32, transposed_objs, is_leaf=is_leaf)
        _class = globals()[obj_type + 's']
        objs = _class(**kwargs)
        check_type(obj_type, objs, _class)
        objects.append(objs)

    return Scene(objects=tuple(objects), camera=camera), view_size


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene dict for expected format (see top of file)'''
    for argname in ('width', 'height'):
        check_type(argname, scene_dict.get(argname), int)

    check_dict_fields(scene_dict.get('Camera'), Camera)

    for outer_obj_dict in scene_dict.get('Objects'):
        for obj_type, obj in outer_obj_dict.items():
            check_dict_fields(obj, globals()[obj_type + 's'])


def check_dict_fields(obj: dict, cls: type) -> None:
    type_hints = get_type_hints(cls)

    check_type('obj', obj, dict)

    provided_fields = set(obj.keys())
    required_fields = set(type_hints.keys())
    if provided_fields != required_fields:
        raise ValueError(
            f'{obj} must have fields {required_fields}, not {provided_fields}'
        )

    for field in obj:
        # cast lists to tuples to be able to check length
        value = tuple(obj[field]) if isinstance(obj[field], list) else obj[field]
        check_type(field, value, UNFLATTENED_TYPES[type_hints[field]])


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)

    n = 10
    pos = np.zeros((n, 3))
    check_type('pos', pos, Vec3s)
    radius = np.zeros((n))
    check_type('radius', radius, Scalars)
    color = np.zeros((n, 3))
    check_type('color', color, Vec3s)
    obj = Spheres(position=pos, radius=radius, color=color)
    check_type('obj', obj, Spheres)

    check_scene_dict(scene_dict)
    scene, view_size = get_scene(scene_dict)
    check_type('scene', scene, Scene)
