from jax import numpy as np, tree_map
from raymarch import Scene, Camera, OBJECT_IDX

# typing
from typeguard import check_type, typechecked
from typing import Tuple, TypedDict, Dict, Any, Sequence

DictStr = Dict[str, Any]


class CameraDict(TypedDict):
    position: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float


@typechecked
def add_obj_dict_defaults(
    attribute: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0, 0, 0),
    position: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    mirror: Tuple[float, float, float] = (0, 0, 0),
    rounding: float = 0,
) -> DictStr:
    return locals()


def is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and all(isinstance(i, (int, float)) for i in x)


def cast_to_tuple(x: Any) -> Any:
    return tuple(x) if is_seq(x) else x


def tree_cast_to_tuple(x: Any) -> Any:
    return tree_map(cast_to_tuple, x, is_leaf=is_seq)


def build_scene(scene_dict: DictStr) -> DictStr:
    '''Create a scene, camera and other parameters from a dict of expected format (see scene.yml)'''

    view_size = scene_dict['height'], scene_dict['width']
    smoothing = scene_dict['smoothing']
    light_dir = scene_dict['light_dir']
    camera_dict = scene_dict['Camera']

    def process(outer_obj_dict: Dict[str, DictStr]) -> Tuple[int, DictStr]:
        obj_name, obj_dict = next(iter(outer_obj_dict.items()))  # get first key-value pair
        return OBJECT_IDX[obj_name], add_obj_dict_defaults(**tree_cast_to_tuple(obj_dict))

    objects, obj_dicts = zip(*[process(outer_obj_dict) for outer_obj_dict in scene_dict['Objects']])

    object_args = tree_map(lambda *xs: np.float32(xs), *obj_dicts, is_leaf=is_seq)  # transpose tree
    object_args = {k + 's': v for k, v in object_args.items()}  # add 's' to pluralize keys
    object_args['mirrorings'] = object_args.pop('mirrors').astype(np.bool_)  # convert mirrorings

    return {
        'scene': Scene(
            objects=np.uint8(objects),
            **object_args,
            smoothing=np.float32(smoothing),
        ),
        'camera': Camera(**tree_map(np.float32, camera_dict, is_leaf=is_seq)),
        'view_size': view_size,
        'light_dir': np.float32(light_dir),
    }


def check_scene_dict(scene_dict: DictStr) -> None:
    '''Check a scene dict for expected format (see scene.yml)'''
    for argname in ('height', 'width'):
        check_type(argname, scene_dict.get(argname), int)
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    check_type('smoothing', scene_dict.get('smoothing'), float)
    check_type('light_dir', cast_to_tuple(scene_dict.get('light_dir')), Tuple[float, float, float])
    check_type('Camera', tree_cast_to_tuple(scene_dict.get('Camera')), CameraDict)

    for outer_obj_dict in scene_dict.get('Objects'):
        obj_name, obj_dict = next(iter(outer_obj_dict.items()))
        assert obj_name in OBJECT_IDX, f'Unknown object name {obj_name}'
        add_obj_dict_defaults(**tree_cast_to_tuple(obj_dict))


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
        check_type('light_dir', out['light_dir'], np.ndarray)
        print('Checked', path.name)
