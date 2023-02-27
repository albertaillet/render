# Raymarching in JAX

Dash app to interact with raymarching in JAX. Toy project to learn about raymarching.

Here are two examples of rendered scenes, a bunch of spheres and a strange snowman:
<p float="left">
<img src="assets/spheres.png" width="45%" title="Spheres">
<img src="assets/snowman.png" width="45%" title="A strange snowman">
</p>

While in the app, the scene lighting can be changed by clicking on the rendered scene.

The scene can also be modified while in the app, it is represented in a yaml format. Here is an example of a scene with a sphere and a plane:

```yaml
width: 200
height: 200
Camera:
    position: [3, 5, 3]
    target: [0, 0, 0]
    up: [0, 1, 0]
Objects:
    - Sphere:
        position: [0, 2, 0]
        radius: 0.5
        color: [0, 0, 1]
    - Plane:
        position: [0, 0, 0]
        normal: [0, 1, 0]
        color: [1, 1, 1]
```

Try it out using colab:

<a href="https://colab.research.google.com/github/albertaillet/render/blob/main/colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

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

- Simple 3D visualization with JAX raycasting, Alexander Mordvintsev, September 23, 2022 ([link](
    https://google-research.github.io/self-organising-systems/2022/jax-raycast/))

- Differentiable Path Tracing on the GPU/TPU, Eric Jang, November 28, 2019 ([link](
    https://blog.evjang.com/2019/11/jaxpt.html), [code](
        https://github.com/ericjang/pt-jax))

