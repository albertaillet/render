'''Build Scene for raymarching from a dict
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
import raymarch as rm
from jax import tree_map, numpy as np
from collections import defaultdict

# typing
from typeguard import check_type
from typing import Tuple, Any, get_type_hints

UNFLATTENED_TYPES = {
    rm.Vec3: Tuple[float, float, float],
    rm.Vec3s: Tuple[float, float, float],
    rm.Scalars: float,
}


def get_class(obj_type: str) -> Any:
    '''Get the class for an object type'''
    return getattr(rm, obj_type + 's')


def get_scene(scene_dict: dict) -> Tuple[rm.Scene, Tuple[int, int]]:
    '''Create a scene from a dict of expected format (see top of file)'''

    def is_leaf(node: Any) -> bool:
        return isinstance(node, list)

    view_size = scene_dict['width'], scene_dict['height']
    camera_dict = scene_dict['Camera']
    objects_dict = scene_dict['Objects']

    camera = rm.Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    object_dicts = defaultdict(list)
    for outer_obj_dict in objects_dict:
        for obj_type, obj in outer_obj_dict.items():  # there should only be one key
            object_dicts[obj_type].append(obj)

    # pad with objects to powers of 2 to avoid recompilation of jitted function
    for obj_type, objs in object_dicts.items():
        while np.log2(len(objs)) % 1 != 0:
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
        _class = get_class(obj_type)
        objs = _class(**kwargs)
        check_type(obj_type, objs, _class)
        objects.append(objs)

    return rm.Scene(objects=tuple(objects), camera=camera), view_size


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene dict for expected format (see top of file)'''
    for argname in ('width', 'height'):
        check_type(argname, scene_dict.get(argname), int)
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    check_dict_fields(scene_dict.get('Camera'), rm.Camera)

    for outer_obj_dict in scene_dict.get('Objects'):
        for obj_type, obj in outer_obj_dict.items():
            check_dict_fields(obj, get_class(obj_type))


def check_dict_fields(obj: dict, cls: type) -> None:
    type_hints = get_type_hints(cls)

    check_type('obj', obj, dict)

    provided = set(obj.keys())
    required = set(type_hints.keys())
    if provided != required:
        raise ValueError(f'{obj} has {provided} fields and should have {required}')

    for field in obj:
        # cast lists to tuples to be able to check length
        value = tuple(obj[field]) if isinstance(obj[field], list) else obj[field]
        check_type(field, value, UNFLATTENED_TYPES[type_hints[field]])


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)

    n = 10
    pos = np.zeros((n, 3))
    check_type('pos', pos, rm.Vec3s)
    radius = np.zeros((n))
    check_type('radius', radius, rm.Scalars)
    color = np.zeros((n, 3))
    check_type('color', color, rm.Vec3s)
    obj = rm.Spheres(position=pos, radius=radius, color=color)
    check_type('obj', obj, rm.Spheres)

    check_scene_dict(scene_dict)
    scene, view_size = get_scene(scene_dict)
    check_type('scene', scene, rm.Scene)
