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

# typing
from typeguard import check_type
from typing import Tuple, Any, get_type_hints

CAST_TYPES = {
    rm.Vec3: Tuple[float, float, float],
    rm.Scalar: float,
}


def get_class(obj_type: str) -> Any:
    '''Get the class for an object type'''
    return getattr(rm, obj_type)


def build_scene(scene_dict: dict) -> Tuple[rm.Scene, Tuple[int, int]]:
    '''Create a scene from a dict of expected format (see top of file)'''

    def is_leaf(node: Any) -> bool:
        return isinstance(node, list)

    view_size = scene_dict['height'], scene_dict['width']
    camera_dict = scene_dict['Camera']
    object_dict_list = scene_dict['Objects']

    camera = rm.Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    objects = []
    for obj_dict in object_dict_list:
        obj_type, obj = next(iter(obj_dict.items()))
        cls = get_class(obj_type)
        objects.append(cls(**tree_map(np.float32, obj, is_leaf=is_leaf)))

    return rm.Scene(objects=tuple(objects), camera=camera), view_size


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene dict for expected format (see top of file)'''
    for argname in ('height', 'width'):
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
        check_type(field, value, CAST_TYPES[type_hints[field]])


if __name__ == '__main__':
    from yaml import SafeLoader, load

    scene_dict = load(open('scenes/scene.yml', 'r'), SafeLoader)

    pos = np.zeros((3))
    check_type('pos', pos, rm.Vec3)
    radius = np.array(0.5)
    check_type('radius', radius, rm.Scalar)
    color = np.zeros((3))
    check_type('color', color, rm.Vec3)
    obj = rm.Sphere(position=pos, radius=radius, color=color)
    check_type('obj', obj, rm.Sphere)

    check_scene_dict(scene_dict)
    scene, view_size = build_scene(scene_dict)
    check_type('scene', scene, rm.Scene)
