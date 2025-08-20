import gradio as gr
from yaml import safe_load
from pathlib import Path
from typing import Tuple, Any

from utils.plot import to_rgb
from raymarch import RenderedImages, render_scene
from builder import check_scene_dict, build_scene
from functools import lru_cache
from numpy.typing import ArrayLike

SCENES_PATH = Path(__file__).parent / 'scenes'
DEFAULT_SCENE_PATH = SCENES_PATH / 'airpods.yml'
scene_choices = [p.stem for p in sorted(SCENES_PATH.glob('*.yml'))]
file_load = lambda stem: (SCENES_PATH / f'{stem}.yml').read_text()  # noqa: E731


def init_app() -> Tuple[str, ArrayLike]:
    scene_str = DEFAULT_SCENE_PATH.read_text()
    view = RenderedImages._fields[0]
    img = render_view(view=view, scene_str=scene_str)
    return scene_str, img


def validate_scene(scene_str: str) -> dict[str, Any]:
    """Validate scene YAML. If valid hide error box else show error details."""
    try:
        check_scene_dict(safe_load(scene_str))
        return gr.update(visible=False, value='')
    except Exception as e:
        return gr.update(visible=True, value=f'**{type(e).__name__}**\n\n```\n{e}\n```')


@lru_cache
def render(scene_str: str, click: Tuple[int, int]) -> RenderedImages:
    """Render the scene and return all the images."""
    scene_dict = check_scene_dict(safe_load(scene_str))
    scene = build_scene(scene_dict)
    return render_scene(**scene, click=click)


def render_view(view: str, scene_str: str, click: Tuple[int, int] = (-1, -1)) -> ArrayLike:
    """Render the scene and return the view."""
    images = render(scene_str=scene_str, click=click)
    image = getattr(images, view)
    return to_rgb(image)


def on_image_click(view: str, scene_str: str, evt: gr.SelectData) -> ArrayLike:
    return render_view(view=view, scene_str=scene_str, click=tuple(evt.index[::-1]))


with gr.Blocks(title='JAX Raymarching (Gradio)') as demo:
    gr.Markdown('## JAX Raymarching')
    with gr.Row():
        scene_dropdown = gr.Radio(
            choices=scene_choices,
            value=DEFAULT_SCENE_PATH.stem,
            label='Load Scene from File',
            interactive=True,
        )

    fields = RenderedImages._fields
    view_radio = gr.Radio(choices=fields, value=fields[0], label='View', interactive=True)

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            code = gr.Code(label='Scene data (YAML)', language='yaml', lines=30)
            error_box = gr.Markdown(visible=False)
        with gr.Column(scale=8):
            img = gr.Image(label='Render', type='numpy', interactive=False)

    demo.load(fn=init_app, inputs=None, outputs=[code, img])

    # Selecting a file fills the editor (validation happens on code.change)
    scene_dropdown.change(fn=file_load, inputs=[scene_dropdown], outputs=[code])

    # Validate YAML on edit, on success, re-render
    code.change(fn=validate_scene, inputs=[code], outputs=[error_box]).then(
        fn=render_view, inputs=[view_radio, code], outputs=[img]
    )

    # Changing the view triggers a re-render
    view_radio.change(fn=render_view, inputs=[view_radio, code], outputs=[img])

    # Clicking on the image triggers a re-render with click=(y,x)
    img.select(fn=on_image_click, inputs=[view_radio, code], outputs=[img])

if __name__ == '__main__':
    demo.launch()
