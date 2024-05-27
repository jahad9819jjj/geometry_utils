import torch
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor
import time
import numpy as np

from ot_downsampling import sampling_farthest_points
from visualization import visualize_principal_curvature, show_mesh_with_pd

Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.5, truncate=1)
def OT_registration(source, target, name, use_cuda=False):
    
    # 各点と各面積をつかって最適化
    a, x = source  # weights, locations
    b, y = target  # weights, locations
    x.requires_grad = True
    z = x.clone()  # Moving point cloud
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    nits = 4 
    for it in range(nits):
        wasserstein_zy = Loss(a, z, b, y)
        [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
        z -= grad_z / a[:, None]  # Apply the regularized Brenier map
        # save_vtk(f"matching_{name}_it_{it}.vtk", numpy(z), colors)
    end = time.time()
    print("Registered {} in {:.3f}s.".format(name, end - start))
    return z

def get_barycenter_from_triangles(verts, triangles):
    A, B, C = verts[triangles[:, 0]], verts[triangles[:, 1]], verts[triangles[:, 2]]
    X = (A + B + C) / 3
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, axis=1)) / 2
    return S / np.sum(S), X

def get_barycenter_from_triangles_extend(triangles, *args):
    barycenters = []
    weights = None
    for arg in args:
        A, B, C = arg[triangles[:, 0]], arg[triangles[:, 1]], arg[triangles[:, 2]]
        X = (A + B + C) / 3
        if weights is None:
            S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, axis=1)) / 2
            weights = S / np.sum(S)
        barycenters.append(X)
    return weights, barycenters

def normalize_arguments(*args):
    normalized_args = []
    for arg in args:
        arg_min = arg.min()
        arg_max = arg.max()
        arg_norm = (arg - arg_min) / (arg_max - arg_min)
        normalized_args.append(arg_norm)
    return normalized_args

if __name__ == "__main__":
    import igl
    import open3d as o3d
    mps_device = torch.device("mps")
    src_path = '/Users/jinhirai/Downloads/Dataset/Mug/Mug4_remesh.ply' #annotation
    trg_path = '/Users/jinhirai/Downloads/Dataset/Mug/Mug10_remesh.ply' #target
    # src_verts, src_triangles = igl.read_triangle_mesh(src_path)
    # trg_verts, trg_triangles = igl.read_triangle_mesh(trg_path)
    src_mesh = o3d.io.read_triangle_mesh(src_path)
    trg_mesh = o3d.io.read_triangle_mesh(trg_path)
    src_verts = np.asarray(src_mesh.vertices); src_colors = np.asarray(src_mesh.vertex_colors); src_triangles = np.asarray(src_mesh.triangles)
    trg_verts = np.asarray(trg_mesh.vertices); trg_colors = np.asarray(trg_mesh.vertex_colors); trg_triangles = np.asarray(trg_mesh.triangles)
    
    src_d1, src_d2, src_v1, src_v2 = igl.principal_curvature(src_verts, src_triangles, radius=5, use_k_ring=True)
    trg_d1, trg_d2, trg_v1, trg_v2 = igl.principal_curvature(trg_verts, trg_triangles, radius=5, use_k_ring=True)
    src_v3 = (src_v1 + src_v2) / 2
    trg_v3 = (trg_v1 + trg_v2) / 2
    src_d3 = np.cross(src_d1, src_d2)
    trg_d3 = np.cross(trg_d1, trg_d2)
    
    # src_weights, src_verts = get_barycenter_from_triangles(src_verts, src_triangles)
    # trg_weights, trg_verts = get_barycenter_from_triangles(trg_verts, trg_triangles)
    src_weights, [src_verts, src_d1, src_d2, src_d3, src_v1, src_v2, src_v3] = get_barycenter_from_triangles_extend(src_triangles, src_verts, src_d1, src_d2, src_d3, src_v1, src_v2, src_v3)
    trg_weights, [trg_verts, trg_d1, trg_d2, trg_d3, trg_v1, trg_v2, trg_v3] = get_barycenter_from_triangles_extend(trg_triangles, trg_verts, trg_d1, trg_d2, trg_d3, trg_v1, trg_v2, trg_v3)
    
    src_v1_norm, src_v2_norm, src_v3_norm = normalize_arguments(src_v1, src_v2, src_v3)
    trg_v1_norm, trg_v2_norm, trg_v3_norm = normalize_arguments(trg_v1, trg_v2, trg_v3)
    
    sample_indices = sampling_farthest_points(src_verts, npoint=500)
    src_msh, src_vf1, src_vf2, src_vf3 = visualize_principal_curvature(src_verts, src_triangles, 
                                                       src_d1, src_v1, 
                                                       src_d2, src_v2, 
                                                       src_d3, src_v3,
                                                       indices=sample_indices,
                                                       scale=0.005
                                                       )
    trg_msh, trg_vf1, trg_vf2, trg_vf3 = visualize_principal_curvature(trg_verts, trg_triangles,
                                                                       trg_d1, trg_v1,
                                                                       trg_d2, trg_v2,
                                                                       trg_d3, trg_v3,
                                                                       indices=sample_indices,
                                                                       scale=0.005
                                                                        )
    show_mesh_with_pd(src_msh, src_vf1, src_vf2, src_vf3)
    show_mesh_with_pd(trg_msh, trg_vf1, trg_vf2, trg_vf3)
    
    src_verts = torch.Tensor(src_verts)
    trg_verts = torch.Tensor(trg_verts)
    src_weights = torch.Tensor(src_weights)
    trg_weights = torch.Tensor(trg_weights)
    
    matching = OT_registration([src_weights, src_verts],
                               [trg_weights, trg_verts],
                               "shape_0")
    raise