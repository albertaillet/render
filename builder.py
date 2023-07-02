from jax import numpy as np, tree_map
from raymarch import Scene, Camera, OBJECT_IDX

# typing
from inspect import signature
from typeguard import check_type, typechecked
from typing import Tuple, Dict, Any

CAMERA_FIELDS = {
    'position': Tuple[float, float, float],
    'target': Tuple[float, float, float],
    'up': Tuple[float, float, float],
    'fov': float,
}


@typechecked
def create_obj_dict(
    attribute: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0, 0, 0),
    position: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    mirror: Tuple[float, float, float] = (0, 0, 0),
    rounding: float = 0,
) -> Dict[str, Any]:
    return locals()


def cast_to_tuple(x: Any) -> Any:
    return tuple(x) if isinstance(x, list) else x


def is_leaf(node: Any) -> bool:
    return isinstance(node, list)


def build_scene(scene_dict: dict) -> Dict[str, Any]:
    '''Create a scene, camera and other parameters from a dict of expected format (see scene.yml)'''

    view_size = scene_dict['height'], scene_dict['width']
    smoothing = scene_dict['smoothing']
    light_dir = scene_dict['light_dir']
    camera_dict = scene_dict['Camera']
    object_dict_list = scene_dict['Objects']

    camera = Camera(**tree_map(np.float32, camera_dict, is_leaf=is_leaf))

    objects = []
    object_args = {arg + 's': [] for arg in signature(create_obj_dict).parameters}
    for obj_dict in object_dict_list:
        obj_type, obj = next(iter(obj_dict.items()))
        objects.append(OBJECT_IDX[obj_type])
        obj_dict = create_obj_dict(**tree_map(cast_to_tuple, obj, is_leaf=is_leaf))
        for arg_name, arg in obj_dict.items():
            object_args[arg_name + 's'].append(arg)

    object_args = tree_map(np.float32, object_args, is_leaf=is_leaf)
    object_args['mirrorings'] = object_args.pop('mirrors').astype(np.bool_)

    return {
        'scene': Scene(
            objects=np.uint8(objects),
            **object_args,
            smoothing=np.float32(smoothing),
        ),
        'camera': camera,
        'view_size': view_size,
        'light_dir': np.float32(light_dir),
    }


def check_scene_dict(scene_dict: dict) -> None:
    '''Check a scene dict for expected format (see scene.yml)'''
    for argname in ('height', 'width'):
        check_type(argname, scene_dict.get(argname), int)
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    check_type('smoothing', scene_dict.get('smoothing'), float)
    check_type('light_dir', cast_to_tuple(scene_dict.get('light_dir')), Tuple[float, float, float])

    check_fields(scene_dict.get('Camera'), CAMERA_FIELDS)

    for outer_obj_dict in scene_dict.get('Objects'):
        for obj_type, obj in outer_obj_dict.items():
            assert obj_type in OBJECT_IDX, f'Unknown object type {obj_type}'
            create_obj_dict(**tree_map(cast_to_tuple, obj, is_leaf=is_leaf))


def check_fields(obj: dict, fields: Dict[str, type]) -> None:
    check_type('obj', obj, dict)

    provided, required = set(obj.keys()), set(fields.keys())
    if provided != required:
        raise ValueError(f'{obj} has {provided} fields and should have {required}')

    for argname, argtype in fields.items():
        # cast lists to tuples to be able to check length
        check_type(argname, cast_to_tuple(obj[argname]), argtype)


if __name__ == '__main__':
    from pathlib import Path
    from utils.plot import load_yaml

    for path in Path('scenes').glob('*.yml'):
        scene_dict = load_yaml(path)
        check_scene_dict(scene_dict)
        out = build_scene(scene_dict)
        check_type('scene', out['scene'], Scene)
        check_type('camera', out['camera'], Camera)
        check_type('view_size', out['view_size'], Tuple[int, int])
        print('Checked', path.name)
