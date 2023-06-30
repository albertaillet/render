# reference: https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_marching_cubes_lewiner.py


def tetrahedron():
    # x, y, z are the coordinates of the grid points
    x = [0, 1, 2, 0]
    y = [0, 0, 1, 2]
    z = [0, 2, 0, 1]

    # i, j and k give the vertices of triangles
    i = [0, 0, 0, 1]
    j = [1, 2, 3, 2]
    k = [2, 3, 1, 3]

    # fc gives the color of the triangles
    fc = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0.5, 0]]

    return {
        'x': x,
        'y': y,
        'z': z,
        'i': i,
        'j': j,
        'k': k,
        'facecolor': fc,
    }


if __name__ == '__main__':
    from builder import build_scene
    from utils.plot import load_yaml, go

    scene, _ = build_scene(load_yaml('scenes/sphere.yaml'))

    go.Figure(go.Mesh3d(**tetrahedron())).show()
