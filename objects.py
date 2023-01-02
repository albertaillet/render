from jax import numpy as np
from utils import norm, softmax

# typing
from jax import Array
from typing import NamedTuple


class Spheres(NamedTuple):
    pos: Array
    radii: Array
    color: Array

    def sdf(self, p: Array) -> Array:
        return norm(p - self.pos) - self.radii


class Planes(NamedTuple):
    pos: Array
    normal: Array
    color: Array

    def sdf(self, p: Array) -> Array:
        return np.einsum('i j, i j -> i', p - self.pos, self.normal)


class Scene(NamedTuple):
    spheres: Spheres
    planes: Planes

    def sdf(self, p: Array) -> Array:
        sphere_dists = self.spheres.sdf(p)
        plane_dists = self.planes.sdf(p)
        return np.concatenate([sphere_dists, plane_dists])
    
    def color(self, p: Array) -> Array:
        colors = np.concatenate([self.spheres.color, self.planes.color])
        dists = self.sdf(p)
        return softmax(-8.0 * dists) @ colors


def get_scene(scene_json: dict) -> Scene:
    '''Create a scene from a JSON object of the form:
    [
        {
            'type': 'Sphere',
            'pos': [0, 0, 0],
            'radius': 1
        },
        {
            'type': 'Plane',
            'pos': [0, 0, 0],
            'normal': [0, 1, 0]
        }
    ]
    '''
    spheres = []
    planes = []
    for obj in scene_json:
        if obj['type'] == 'Sphere':
            spheres.append(obj)
        elif obj['type'] == 'Plane':
            planes.append(obj)
        else:
            e = obj['type']
            raise ValueError(f'Unknown object type: {e}')
    try:
        scene = Scene(
            Spheres(
                pos=np.array([s['position'] for s in spheres]),
                radii=np.array([s['radius'] for s in spheres]),
                color=np.array([s['color'] for s in spheres]),
            ),
            Planes(
                pos=np.array([p['position'] for p in planes]),
                normal=np.array([p['normal'] for p in planes]),
                color=np.array([p['color'] for p in planes]),
            ),
        )
    except KeyError as e:
        raise ValueError(f'Missing key: {e}')
    return scene
