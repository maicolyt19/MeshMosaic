import os

import numpy as np
import trimesh


def normalize_mesh_export(mesh, file_out=None):
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 2.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)


if __name__ == '__main__':
    input_dir = './data/different_parts'
    output_dir = './data/different_parts'
    for f in os.listdir(input_dir):
        if os.path.splitext(f)[1] in ['.obj']:
            normalize_mesh_export(trimesh.load(os.path.join(input_dir, f), process=False), os.path.join(input_dir, f))
