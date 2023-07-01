# Raymarching in JAX

Dash app to interact with raymarching in JAX. Toy project to learn about raymarching.

Try it out using colab:

<a href="https://colab.research.google.com/github/albertaillet/render/blob/main/colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Here are three examples of rendered scenes and their corresponding normals:
<p float="left">
<img src="assets/cornell_box.png" width="32%" title="Cornell_box">
<img src="assets/snowman.png" width="32%" title="A strange snowman">
<img src="assets/airpods.png" width="32%" title="Spheres">
</p>
<p float="left">
<img src="assets/cornell_box_normals.png" width="32%" title="A strange snowman">
<img src="assets/snowman_normals.png" width="32%" title="Spheres">
<img src="assets/airpods_normals.png" width="32%" title="A strange snowman">
</p>

While in the app, the scene lighting can be changed by clicking on the rendered scene.

The scene can also be modified while in the app, it is represented in a yaml format. Here is an example of a scene with a sphere, a box and a plane:

```yaml
width: 200
height: 200
smoothing: 0.07
Camera:
    position: [1, 1, 0]
    target: [0, 0.5, 0]
    up: [0, 1, 0]
    fov: 0.6
Objects:
    - Sphere:
        position: [0, 0.7, 0]
        attribute: [0.1, 0, 0]
        rotation: [6, 1, 0]
        color: [1, 1, 0.5]
    - Box:
        position: [0, 0.3, 0]
        attribute: [0.1, 0.2, 0.3]
        rotation: [0, 0.5, 0]
        color: [1, 0.4, 1]
        rounding: 0.04
    - Plane:
        attribute: [0, 1, 0]
        color: [1, 1, 1]
```

## Usage

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

## References

- Simple 3D visualization with JAX raycasting, Alexander Mordvintsev, September 23, 2022 ([post](
    https://google-research.github.io/self-organising-systems/2022/jax-raycast/), [code](https://github.com/google-research/self-organising-systems/blob/master/notebooks/jax_raycast.ipynb))

- Differentiable Path Tracing on the GPU/TPU, Eric Jang, November 28, 2019 ([post](
    https://blog.evjang.com/2019/11/jaxpt.html), [code](
        https://github.com/ericjang/pt-jax))

- 3D Signed Distance Functions, Inigo Quilez ([post](
    https://iquilezles.org/articles/distfunctions/), [code](
        https://www.shadertoy.com/view/Xds3zN))
