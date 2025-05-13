import glob
import os
import math
import torch
import numpy as np
import skimage.feature
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ####import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras, RasterizationSettings, MeshRasterizer,
                                TexturesVertex, PointLights, MeshRenderer, HardPhongShader, Materials, HardFlatShader)
from pytorch3d.renderer.mesh.shader import (BlendParams)


def calculate_target_coordinates(r, phi_1, phi_2):
    phi_1 = np.deg2rad(phi_1)
    phi_2 = np.deg2rad(phi_2)
    x = r * math.sin(phi_1)*math.cos(phi_2)
    y = r * math.sin(phi_1)*math.sin(phi_2)
    z = r * math.cos(phi_1)
    return x, y, z


# ###make all face towards consistency
def make_front_faces(mesh_pytorch3d):

    # compute the normals for the mesh
    verts = mesh_pytorch3d.verts_packed()
    faces = mesh_pytorch3d.faces_packed().float()
    textures = mesh_pytorch3d.textures
    face_normals = mesh_pytorch3d._faces_normals_packed

    # convert the normals to face normals
    # face_normals = normals.view(num_faces, 3, 3).mean(dim=1)

    # get the dot product between the normals and the faces
    dot_product = (face_normals * faces.norm(dim=1, keepdim=True)).sum(dim=1)

    # create a mask for the front-facing and back-facing triangles
    back_facing_mask = dot_product < 0

    # reverse the order of vertices for the back-facing triangles
    faces[back_facing_mask] = torch.flip(faces[back_facing_mask], [1])
    faces = faces.long()

    # update the mesh with the new faces
    mesh_pytorch3d = Meshes(verts=[verts], faces=[faces], textures=textures)

    # return the mesh with front-facing triangles
    return mesh_pytorch3d


# ####nomoralize the point cloud
def normalize_point(points):
    # ####normalize point xyz to [0, 1]
    center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    dists = np.sqrt(np.sum((points - center) ** 2, axis=1))
    max_dist = np.max(dists)
    points /= max_dist

    # ####move the center points to the origin
    center = np.mean([np.min(points, axis=0), np.max(points, axis=0)], axis=0)
    points = points - center
    return points


def get_npy_from_off(off_path, save_path):
    with open(off_path, 'r') as f:
        f.readline()  # skip the file head
        num_line =  f.readline()
        num_vert, num_face, _ = map(int, num_line.split())

        points = []
        faces = []
        # ####read location
        for l_i in range(num_vert):
            cur_line = f.readline()
            x, y, z = map(float, cur_line.split())
            points.append([x, y, z])

        # ####read faces
        for l_i in range(num_face):
            cur_line = f.readline()
            _, face_0, face_1, face_2 = map(int, cur_line.split())
            faces.append([face_0, face_1, face_2])

        points = np.array(points)
        faces = np.array(faces)
        points = normalize_point(points)
        f.close()
    np.save(save_path + '_xyz.npy', points)
    np.save(save_path + '_faces.npy', faces)
    pass


# #### get data from off file for ModelNet10
def get_data_from_off(test_file_list, save_dir):
    for f_i, f_path in enumerate(test_file_list):
        file_name = os.path.splitext(os.path.basename(f_path))[0]
        save_path = os.path.join(save_dir, file_name)
        get_npy_from_off(f_path, save_path)
        print('complete {}/{}: {}'.format(f_i, len(test_file_list), file_name))
    pass



if __name__ == "__main__":
    test_file_list = glob.glob('./ModelNet10/ModelNet10/*/test/*.off')
    test_file_list.sort()
    save_npy_dir = 'ModelNet10/npy_render'
    os.makedirs(save_npy_dir, exist_ok=True)
    get_data_from_off(test_file_list, save_npy_dir)