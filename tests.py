from sys import argv
from yaml import safe_load
from utils.plot import to_rgb
from typeguard import check_type
from matplotlib import pyplot as plt
from raymarch import render_scene, RenderedImages
from builder import build_scene, check_scene_dict, SceneDict
from app import SCENES_PATH


def test_all_scenes():
    annotations = render_scene.__annotations__
    checked_args = ('objects', 'camera', 'view_size', 'light_dir')
    for path in SCENES_PATH.glob('*.yml'):
        scene_dict = check_type(check_scene_dict(safe_load(open(path))), SceneDict)
        scene = build_scene(scene_dict)
        for arg in checked_args:
            check_type(scene[arg], annotations[arg])
        print('Checked', path.stem)


def test_rendering(scene_name: str) -> None:
    scene_file = SCENES_PATH / (scene_name + '.yml')
    scene = build_scene(check_scene_dict(safe_load(open(scene_file))))
    images = check_type(render_scene(**scene, click=(-1, -1)), RenderedImages)
    plot_rendered_images(images)


def plot_rendered_images(images: RenderedImages) -> None:
    image_names = images._fields
    rows, cols = 3, 3
    plt.style.use('grayscale')
    _, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for ax, name in zip(axs.flatten(), image_names):
        ax.imshow(to_rgb(getattr(images, name)))
        ax.set_title(name.capitalize())
    for ax in axs.flatten():
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(argv) == 1:
        test_all_scenes()  # if no arguments, test build_scene on all scenes
    elif len(argv) == 2:
        test_rendering(argv[1])  # if one argument, test rendering that scene
    else:
        raise ValueError('Invalid number of arguments')
