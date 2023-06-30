'''Build Scene for raymarching from a dict
Expected scene dict format:
{
    'width': 800
    'height': 600
    'smoothing': 0.125
    'Camera': {
        'position': [0, 0, 5]
        'target': [0, 0, 0]
        'up': [0, 1, 0]
        'f': 0.6
    }
    'Objects': [
        {
            'Sphere': {
                'position': [0, 0, 0],
                'attribute': [0.5, 0, 0],
                'color': [0, 0, 1],
                'rotation': [0, 0, 0],
                'rounding': 0.0
            }
        },
        {
            'Plane': {
                'position': [0, 0, 0],
                'attribute': [0, 1, 0],
                'color': [1, 1, 1]
                'rotation': [0, 0, 0],
                'rounding': 0.0
            }
        },
    ]
}
'''
import raymarch as rm
from jax import tree_map, numpy as np

# typing
from typeguard import check_type
from typing import Dict, Tuple, Any, Union

CAMERA_FIELDS = {
    'position': Tuple[float, float, float],
    'target': Tuple[float, float, float],
    'up': Tuple[float, float, float],
    'fov': float,
}
OBJECT_FIELDS = {
    'position': Tuple[float, float, float],
    'attribute': Tuple[float, float, float],
    'color': Tuple[float, float, float],
    'rotation': Tuple[float, float, float],
    'rounding': float,
}


def is_leaf(node: Any) -> bool:
    return isinstance(node, list)


def build_scene(scene_dict: dict) -> Dict[str, Union[rm.Scene, rm.Camera, Tuple[int, int]]]:
    '''Create a scene from a dict of expected format (see top of file)'''

    view_size = scene_dict['height'], scene_dict['width']
    smoothing = scene_dict['smoothing']
    camera_dict = scene_dict['Camera']
    object_dict_list = scene_dict['Objects']

    camera = rm.Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    objects = []
    object_args = {arg + 's': [] for arg in OBJECT_FIELDS}
    for obj_dict in object_dict_list:
        obj_type, obj = next(iter(obj_dict.items()))
        idx = rm.OBJECT_IDX[obj_type]
        objects.append(idx)
        for arg in OBJECT_FIELDS:
            object_args[arg + 's'].append(obj[arg])

    return {
        'scene': rm.Scene(
            objects=np.uint8(objects),
            **tree_map(np.float32, object_args, is_leaf=is_leaf),
            smoothing=np.float32(smoothing),
        ),
        'camera': camera,
        'view_size': view_size,
    }


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene dict for expected format (see top of file)'''
    for argname in ('height', 'width'):
        check_type(argname, scene_dict.get(argname), int)
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    check_type('smoothing', scene_dict.get('smoothing'), float)

    check_fields(scene_dict.get('Camera'), CAMERA_FIELDS)

    for outer_obj_dict in scene_dict.get('Objects'):
        for obj_type, obj in outer_obj_dict.items():
            assert obj_type in rm.OBJECT_IDX, f'Unknown object type {obj_type}'
            check_fields(obj, OBJECT_FIELDS)


def check_fields(obj: dict, fields: Dict[str, type]) -> None:
    check_type('obj', obj, dict)

    provided = set(obj.keys())
    required = set(fields.keys())
    if provided != required:
        raise ValueError(f'{obj} has {provided} fields and should have {required}')

    for argname, argtype in fields.items():
        # cast lists to tuples to be able to check length
        arg = obj[argname]
        arg = tuple(arg) if isinstance(arg, list) else arg
        check_type(argname, arg, argtype)


if __name__ == '__main__':
    from pathlib import Path
    from utils.plot import load_yaml

    for path in Path('scenes').glob('*.yml'):
        print(f'Checking {path}')
        scene_dict = load_yaml(path)
        check_scene_dict(scene_dict)
        out = build_scene(scene_dict)
        check_type('scene', out['scene'], rm.Scene)
        check_type('camera', out['camera'], rm.Camera)
        check_type('view_size', out['view_size'], Tuple[int, int])
