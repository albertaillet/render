from jax import numpy as np, tree_map
from raymarch import Objects, Camera, OBJECTS

# typing
from typeguard import check_type
from typing import Tuple, TypedDict, Sequence, Any, Dict, List


class CameraDict(TypedDict):
    position: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov: float


class ObjectDict(TypedDict):
    attribute: Tuple[float, float, float]
    color: Tuple[float, float, float]
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    mirror: Tuple[int, int, int]
    rounding: float


class SceneDict(TypedDict):
    height: int
    width: int
    smoothing: float
    light_dir: Tuple[float, float, float]
    Camera: CameraDict
    Objects: List[Tuple[str, ObjectDict]]


def add_obj_dict_defaults(
    attribute: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0, 0, 0),
    position: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    mirror: Tuple[int, int, int] = (0, 0, 0),
    rounding: float = 0,
) -> ObjectDict:
    return check_type(locals(), ObjectDict)


def is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and not any(isinstance(i, dict) for i in x)


def check_scene_dict(scene_dict: Dict[str, Any]) -> SceneDict:
    '''Check a scene dict for expected format and add default values where needed'''
    for argname in ('height', 'width'):
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    # cast all lists (that do not contain dicts) to tuples to be able to check length
    scene_dict = tree_map(lambda x: tuple(x) if is_seq(x) else x, scene_dict, is_leaf=is_seq)

    # checking the obj_names and adding default values where needed
    for i in range(len(scene_dict['Objects'])):
        obj_name, obj_dict = next(iter(scene_dict['Objects'][i].items()))  # first item of dict
        assert obj_name in OBJECTS, f'Unknown object name {obj_name}'
        scene_dict['Objects'][i] = (obj_name, add_obj_dict_defaults(**obj_dict))

    return check_type(scene_dict, SceneDict)


def build_scene(scene_dict: SceneDict) -> Dict[str, Any]:
    '''Create a scene, camera and other parameters from a dict of expected format (see scene.yml)'''
    obj_names, obj_dicts = zip(*scene_dict['Objects'])
    obj_args = tree_map(lambda *xs: xs, *obj_dicts, is_leaf=is_seq)  # transpose tree
    obj_args = {k + 's': v for k, v in obj_args.items()}  # add 's' to pluralize keys
    obj_args = tree_map(np.float32, obj_args, is_leaf=is_seq)  # convert to float32
    obj_args['mirrorings'] = obj_args.pop('mirrors').astype(np.bool_)  # convert mirrorings
    return {
        'objects': Objects(
            ids=np.uint8([OBJECTS.index(o) for o in obj_names]),  # type: ignore
            **obj_args,
            smoothing=np.float32(scene_dict['smoothing']),
        ),
        'camera': Camera(**tree_map(np.float32, scene_dict['Camera'], is_leaf=is_seq)),
        'view_size': (scene_dict['height'], scene_dict['width']),
        'light_dir': np.float32(scene_dict['light_dir']),
    }
