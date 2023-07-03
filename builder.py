from jax import numpy as np, tree_map
from raymarch import Scene, Camera, OBJECT_IDX

# typing
from typeguard import typechecked
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
    mirror: Tuple[float, float, float]
    rounding: float


class SceneDict(TypedDict):
    height: int
    width: int
    smoothing: float
    light_dir: Tuple[float, float, float]
    Camera: CameraDict
    Objects: List[Tuple[str, ObjectDict]]


@typechecked
def add_obj_dict_defaults(
    attribute: Tuple[float, float, float],
    color: Tuple[float, float, float] = (0, 0, 0),
    position: Tuple[float, float, float] = (0, 0, 0),
    rotation: Tuple[float, float, float] = (0, 0, 0),
    mirror: Tuple[float, float, float] = (0, 0, 0),
    rounding: float = 0,
) -> ObjectDict:
    return locals()


def is_seq(x: Any) -> bool:
    return isinstance(x, Sequence) and not any(isinstance(i, dict) for i in x)


def build_scene(scene_dict: SceneDict) -> Dict[str, Any]:
    '''Create a scene, camera and other parameters from a dict of expected format (see scene.yml)'''

    view_size = scene_dict['height'], scene_dict['width']
    smoothing = scene_dict['smoothing']
    light_dir = scene_dict['light_dir']
    camera_dict = scene_dict['Camera']
    obj_names, obj_dicts = zip(*scene_dict['Objects'])

    obj_args = tree_map(lambda *xs: np.float32(xs), *obj_dicts, is_leaf=is_seq)  # transpose tree
    obj_args = {k + 's': v for k, v in obj_args.items()}  # add 's' to pluralize keys
    obj_args['mirrorings'] = obj_args.pop('mirrors').astype(np.bool_)  # convert mirrorings

    return {
        'scene': Scene(
            objects=np.uint8([OBJECT_IDX[o] for o in obj_names]),
            **obj_args,
            smoothing=np.float32(smoothing),
        ),
        'camera': Camera(**tree_map(np.float32, camera_dict, is_leaf=is_seq)),
        'view_size': view_size,
        'light_dir': np.float32(light_dir),
    }


@typechecked
def check_scene_dict(scene_dict: Dict[str, Any]) -> SceneDict:
    '''Check a scene dict for expected format (see scene.yml)'''
    for argname in ('height', 'width'):
        assert scene_dict[argname] > 0, f'{argname} must be positive'

    # cast all lists (that do not contain dicts) to tuples to be able to check length
    scene_dict = tree_map(lambda x: tuple(x) if is_seq(x) else x, scene_dict, is_leaf=is_seq)

    # checking the obj_names and adding default values where needed
    for i in range(len(scene_dict['Objects'])):
        obj_name, obj_dict = next(iter(scene_dict['Objects'][i].items()))  # first item of dict
        assert obj_name in OBJECT_IDX, f'Unknown object name {obj_name}'
        scene_dict['Objects'][i] = (obj_name, add_obj_dict_defaults(**obj_dict))

    return scene_dict


if __name__ == '__main__':
    from pathlib import Path
    from utils.plot import load_yaml
    from typeguard import check_type

    for path in Path('scenes').glob('*.yml'):
        scene_dict = load_yaml(path)
        scene_dict = check_scene_dict(scene_dict)
        out = build_scene(scene_dict)
        check_type('scene_dict', scene_dict, SceneDict)
        check_type('scene', out['scene'], Scene)
        check_type('camera', out['camera'], Camera)
        check_type('view_size', out['view_size'], Tuple[int, int])
        check_type('light_dir', out['light_dir'], np.ndarray)
        print('Checked', path.name)
