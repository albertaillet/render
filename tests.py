from sys import argv
from time import time
from pathlib import Path
from numpy import hstack
from yaml import safe_load
from typeguard import check_type
from utils.plot import to_rgb, fromarray, Image
from raymarch import render_scene, RenderedImages
from builder import build_scene, check_scene_dict, SceneDict
from app import SCENES_PATH


def test_build_scene(scene_path: Path) -> None:
    annotations = render_scene.__annotations__
    checked_args = ('objects', 'camera', 'view_size', 'light_dir')
    scene_dict = check_type(check_scene_dict(safe_load(open(scene_path))), SceneDict)
    scene = build_scene(scene_dict)
    for arg in checked_args:
        check_type(scene[arg], annotations[arg])
    print('Checked', scene_path.stem)


def test_rendering(scene_path: Path) -> Image:
    scene = build_scene(check_scene_dict(safe_load(open(scene_path))))
    t = time()
    images = render_scene(**scene, click=(-1, -1))
    print(f'Rendered {scene_path.stem:<15} | {time() - t:.2f} seconds')
    images = check_type(images, RenderedImages)
    return fromarray(hstack([to_rgb(image) for image in images]))


if __name__ == '__main__':
    tmp = Path('tmp')
    tmp.mkdir(exist_ok=True)
    match tuple(argv[1:]):
        case '-b',:  # build all scenes
            for scene_path in SCENES_PATH.glob('*.yml'):
                test_build_scene(scene_path)
        case '-r',:  # render all scenes
            for scene_path in SCENES_PATH.glob('*.yml'):
                test_rendering(scene_path).save(tmp / f'{scene_path.stem}.png')
        case '-r', scene_name:  # render a specific scene
            test_rendering(SCENES_PATH / f'{scene_name}.yml').save(tmp / f'{scene_name}.png')
        case _:
            print('Usage: python tests.py [-b] [-r [scene_name]]')
