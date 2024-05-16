import numpy as np
import pyvista
import open3d as o3d
import point_cloud_utils
import time
import torch
import copy
    
from sklearn.decomposition import PCA

from ot_downsampling import sampling_farthest_points
from unitvec2rotmat import create_rotation_matrix, create_coordinate_frame_mesh, apply_rotation_matrices
import polyscope
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA

from geomloss import SamplesLoss

class CurvatureInfo:
    def __init__(self):
        self.PD1 = None
        self.PD2 = None
        self.PV1 = None
        self.PV2 = None
        self.PV3 = None # normal scale
        self.PD3 = None # normal direction
        self.Curv = None

def load_ply_mesh(filename):
    start = time.time()
    with open(filename, 'r') as file:
        lines = file.read().splitlines()

    vertex_count = 0
    face_count = 0
    header_end_index = 0

    # Read PLY header
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('element face'):
            face_count = int(line.split()[-1])
        elif line.startswith('end_header'):
            header_end_index = i
            break

    # Read vertex data
    vertex_data = np.genfromtxt(lines[header_end_index + 1:header_end_index + 1 + vertex_count], dtype=float)
    verts = vertex_data[:, :3]
    verts_mean = np.mean(verts, axis=0)
    verts -= verts_mean
    
    curv_info = CurvatureInfo()
    curv_info.PD1 = vertex_data[:, 3:6]
    curv_info.PD2 = vertex_data[:, 6:9]
    curv_info.PV1 = vertex_data[:, 9]
    curv_info.PV2 = vertex_data[:, 10]
    curv_info.Curv = vertex_data[:, 11]

    # Read face data
    face_data = np.genfromtxt(lines[header_end_index + 1 + vertex_count:header_end_index + 1 + vertex_count + face_count], dtype=int)[:, 1:]
    faces = face_data

    end = time.time()
    print(f'Elapsed time : {end - start}')
    return verts, faces, curv_info

def compute_average_edge_length(vertices, faces):
    """
    vertices: numpy array of shape (num_vertices, 3)
    faces: numpy array of shape (num_faces, 3)
    """
    # get coordinate of triangles
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # compute edge length of triangles
    edge_lengths = np.concatenate([
        np.linalg.norm(v1 - v2, axis=1),
        np.linalg.norm(v2 - v3, axis=1),
        np.linalg.norm(v3 - v1, axis=1)
    ])
    
    # compute average edge length
    average_edge_length = np.mean(edge_lengths)
    
    return average_edge_length

def triangles2pvfaces(triangles):
    faces = np.concatenate((np.full((triangles.shape[0], 1), 3), triangles), axis=1)
    return faces

def get_pyvista_coord_frame(size=10):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size)
    verts = np.asarray(frame.vertices)
    triangles = np.asarray(frame.triangles)
    # faces = np.concatenate((np.full((triangles.shape[0], 1), 3), triangles), axis=1)
    faces = triangles2pvfaces(triangles)
    frame_pv = pyvista.PolyData(verts, faces)
    frame_pv.point_data["color"] = np.asarray(frame.vertex_colors)
    return frame_pv

def visualize_principal_curvature(verts, faces=None, 
                                  pd1=None, pv1=None, 
                                  pd2=None, pv2=None, 
                                  pd3=None, pv3=None, 
                                  indices=None, scale=None):
    ''' 
    sample_indices = np.random.choice(mesh.n_points, int(mesh.n_points * sample_rate), replace=False)
    visualize_principal_curvature(..., indices=sample_indices)
    '''
    if faces is not None:
        triangles = np.concatenate((np.full((faces.shape[0], 1), 3), faces), axis=1)
        mesh = pyvista.PolyData(verts, triangles)
    else:
        mesh = pyvista.PolyData(verts)
    
    if indices is None:
        indices = np.arange(mesh.n_points)

    mesh.point_data["pd1"] = pd1
    mesh.point_data["pv1"] = pv1
    mesh.point_data["pd2"] = pd2
    mesh.point_data["pv2"] = pv2
    mesh.point_data["normals"] = pd3
    mesh.point_data["normals_scale"] = pv3


    sampled_pcd = mesh.extract_points(indices)

    # Generate vector field of principal curvature
    if scale is None:
        scale = 10
        vf1 = sampled_pcd.glyph(
            orient="pd1", factor=scale, geom=pyvista.Arrow(),
        )
        vf2 = sampled_pcd.glyph(
            orient="pd2", factor=scale, geom=pyvista.Arrow(),
        )
        vf3 = sampled_pcd.glyph(
            orient="normals", factor=scale, geom=pyvista.Arrow(),
        )

    else:
        vf1 = sampled_pcd.glyph(
            orient="pd1", scale="pv1", factor=scale, geom=pyvista.Arrow(),
        )
        vf2 = sampled_pcd.glyph(
            orient="pd2", scale="pv2", factor=scale, geom=pyvista.Arrow(),
        )
        vf3 = sampled_pcd.glyph(
            orient="normals", scale="normals_scale", factor=scale, geom=pyvista.Arrow(),
        )
    return mesh, vf1, vf2, vf3


def show_two_mesh_and_vf(mesh_x, x_vf1, x_vf2, x_vf3, 
                         mesh_y, y_vf1, y_vf2, y_vf3, 
                         x_vf1_comp=None,
                         y_vf1_comp=None,
                         sync_view:bool=True,
                         viz_pca:bool=False):
    # Create a subplot with 2 rows and 1 column
    plotter = pyvista.Plotter(shape=(1, 2))

    # Add the first mesh and vector fields to the first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_x, color='gray', opacity=0.3)
    plotter.add_mesh(x_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(x_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(x_vf3, line_width=2, color='b')
    plotter.add_title("Mesh X")

    if viz_pca and (x_vf1_comp is not None):
        plotter = add_principal_comp(plotter, x_vf1_comp)

    # Add the second mesh and vector fields to the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_y, color='gray', opacity=0.3)
    plotter.add_mesh(y_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(y_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(y_vf3, line_width=2, color='b')
    plotter.add_title("Mesh Y")

    if viz_pca and (y_vf1_comp is not None):
        plotter = add_principal_comp(plotter, y_vf1_comp)

    # Link the camera between subplots
    if sync_view:
        plotter.link_views()

    # Show the plot
    plotter.show()

def principal_analysis(data, weights):
    '''
    Principal analization of Weighted data
    '''
    # 点群とPVだけつかった主成分分析
    # PDとPVだけつかった主成分分析
    # Lie groupかRiemannian上で主成分分析すれば対称性によらず主成分が求まるのでは？

    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    weights_column = weights_norm.reshape(-1, 1)
    weighted_data = data * weights_column
    pca = PCA(n_components=3)
    pca.fit(weighted_data)
    principal_components = pca.components_
    projected_data = pca.transform(weighted_data)

    # principal_components[2] = np.cross(principal_components[0], principal_components[1])
    # PVの分散を見て、第1主成分~第3主成分の方向を決める

    return projected_data, principal_components

def add_principal_comp(plotter, principal_comp):
    '''
    Add principal components(1,2,3) to plotter
    '''
    origin = np.array([[0, 0, 0]])
    for i, pc in enumerate(principal_comp):
        if i == 0:
            color = 'red'
        elif i == 1:
            color = 'green'
        else:
            color = 'blue'
        
        arrow = pyvista.Arrow(origin, pc, shaft_radius=0.02, tip_radius=0.04, tip_length=0.1, scale=100)
        plotter.add_mesh(arrow, color=color)
    return plotter

def show_distribution_lrf(mesh,
                          lrf_x, lrf_y, lrf_z, weight_x, weight_y, weight_z,
                          principal_lrf_x=None, principal_lrf_y=None, principal_lrf_z=None):
    ''' Show distribution of local reference frame.
    lrf_* : (N, 3)
    weight_* : (N, 1)
    principal_lrf_* : (3, 3)
    '''
    transp = 0.2
    cm = 'jet'
    sphere_plotter = pyvista.Plotter()
    sphere_plotter = pyvista.Plotter(shape=(1, 3))
    sphere_plotter.subplot(0, 0)
    sphere_plotter.add_mesh(mesh)
    sphere_plotter.add_points(lrf_x,
                              scalars=weight_x,
                              cmap=cm,
                              opacity=transp
                              )
    sphere_plotter.add_title("Principal Curvature Direction 1")
    sphere_plotter.subplot(0, 1)
    sphere_plotter.add_mesh(mesh)
    sphere_plotter.add_points(lrf_y,
                              scalars=weight_y,
                              cmap=cm,
                              opacity=transp
                              )
    sphere_plotter.add_title("Principal Curvature Direction 2")
    sphere_plotter.subplot(0, 2)
    sphere_plotter.add_mesh(mesh)
    # scales = (X_Curv.PV3 - X_Curv.PV3.min()) / (np.max(X_Curv.PV3) - np.min(X_Curv.PV3))
    sphere_plotter.add_points(lrf_z,
                              scalars=weight_z,
                              cmap=cm,
                              opacity=transp
                              )
    sphere_plotter.add_title("Principal Curvature Direction 3(Normal)")
    if principal_lrf_x is not None:
        sphere_plotter.subplot(0, 0)
        sphere_plotter.add_points(principal_lrf_x[0], point_size=20, color='r', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_x[1], point_size=20, color='g', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_x[2], point_size=20, color='b', render_points_as_spheres=True)
    if principal_lrf_y is not None:
        sphere_plotter.subplot(0, 1)
        sphere_plotter.add_points(principal_lrf_y[0], point_size=20, color='r', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_y[1], point_size=20, color='g', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_y[2], point_size=20, color='b', render_points_as_spheres=True)
    if principal_lrf_z is not None:
        sphere_plotter.subplot(0, 2)
        sphere_plotter.add_points(principal_lrf_z[0], point_size=20, color='r', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_z[1], point_size=20, color='g', render_points_as_spheres=True)
        sphere_plotter.add_points(principal_lrf_z[2], point_size=20, color='b', render_points_as_spheres=True)
    sphere_plotter.link_views()
    return sphere_plotter


def mesh_norm(verts, triangles=None):
    vert_norm = (verts - verts.min()) / (verts.max() - verts.min())
    if triangles is not None:
        model_norm = pyvista.PolyData(vert_norm, triangles2pvfaces(triangles))
    else:
        model_norm = pyvista.PolyData(vert_norm)
    model_norm = model_norm.translate(np.asarray(model_norm.center) * -1)
    return model_norm

def OT_registration(source, target, name):
    use_cuda = True
    Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.5, truncate=1)
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

if __name__ == "__main__":
    # Store principal curv direction 1, 2 and principal curv value 1,2
    # Use debug/curv/*.ply in main/cpp
    # TODO:同じ手法でリメッシュしたモデルで比較
    x_path = "win/VisualStudio/BCPD-Win/debug/curv/X_mesh.ply"
    y_path = "win/VisualStudio/BCPD-Win/debug/curv/Y_mesh.ply"
    X_verts, X_faces, X_Curv = load_ply_mesh(x_path)
    Y_verts, Y_faces, Y_Curv = load_ply_mesh(y_path)

    show_X_verts = copy.copy(X_verts)
    show_X_faces = copy.copy(X_faces)
    show_Y_verts = copy.copy(Y_verts)
    show_Y_faces = copy.copy(Y_faces)



    num_samples = 10000
    face_id, barycen = point_cloud_utils.sample_mesh_random(X_verts, X_faces, num_samples)
    unique_vert_id = np.unique(X_faces[face_id])
    # unique_vert_id = sampling_farthest_points(X_verts, npoint=num_samples)
    X_verts = X_verts[unique_vert_id]
    X_Curv.PD1 = X_Curv.PD1[unique_vert_id]
    X_Curv.PD2 = X_Curv.PD2[unique_vert_id]
    X_Curv.PV1 = X_Curv.PV1[unique_vert_id]
    X_Curv.PV2 = X_Curv.PV2[unique_vert_id]

    face_id, barycen = point_cloud_utils.sample_mesh_random(Y_verts, Y_faces, num_samples)
    unique_vert_id = np.unique(Y_faces[face_id])
    # unique_vert_id = sampling_farthest_points(Y_verts, npoint=num_samples)
    Y_verts = Y_verts[unique_vert_id]
    Y_Curv.PD1 = Y_Curv.PD1[unique_vert_id]
    Y_Curv.PD2 = Y_Curv.PD2[unique_vert_id]
    Y_Curv.PV1 = Y_Curv.PV1[unique_vert_id]
    Y_Curv.PV2 = Y_Curv.PV2[unique_vert_id]

    # compute average edge length but unused yet
    # avg_x_edge_len = compute_average_edge_length(X_verts, X_faces)
    # avg_y_edge_len = compute_average_edge_length(Y_verts, Y_faces)
    avg_x_edge_len = 1
    avg_y_edge_len = 1

    X_Curv.PD3 = np.cross(X_Curv.PD1, X_Curv.PD2)
    X_Curv.PV3 = (X_Curv.PV1 + X_Curv.PV2) / 2
    Y_Curv.PD3 = np.cross(Y_Curv.PD1, Y_Curv.PD2)
    Y_Curv.PV3 = (Y_Curv.PV1 + Y_Curv.PV2) / 2

    X_PV1_norm = (X_Curv.PV1 - X_Curv.PV1.min()) / (X_Curv.PV1.max() - X_Curv.PV1.min())
    X_PV2_norm = (X_Curv.PV2 - X_Curv.PV2.min()) / (X_Curv.PV2.max() - X_Curv.PV2.min())
    X_PV3_norm = (X_Curv.PV3 - X_Curv.PV3.min()) / (X_Curv.PV3.max() - X_Curv.PV3.min())
    Y_PV1_norm = (Y_Curv.PV1 - Y_Curv.PV1.min()) / (Y_Curv.PV1.max() - Y_Curv.PV1.min())
    Y_PV2_norm = (Y_Curv.PV2 - Y_Curv.PV2.min()) / (Y_Curv.PV2.max() - Y_Curv.PV2.min())
    Y_PV3_norm = (Y_Curv.PV3 - Y_Curv.PV3.min()) / (Y_Curv.PV3.max() - Y_Curv.PV3.min())

    # 各点のlrfの分布を算出。ある点を基準とし、8象限に分け、その分布を出す。その分布間の距離を算出し、ある点と類似度が高くなるような計算を行う
    x_mesh, x_vf1, x_vf2, x_vf3 = visualize_principal_curvature(
                                                        X_verts, 
                                                        # X_faces, 
                                                        pd1=X_Curv.PD1, 
                                                        pv1=X_PV1_norm * avg_x_edge_len, 
                                                        pd2=X_Curv.PD2,
                                                        pv2=X_PV2_norm * avg_x_edge_len,
                                                        pd3=X_Curv.PD3,
                                                        pv3=X_PV3_norm * avg_x_edge_len,
                                                        scale=0.1)
    y_mesh, y_vf1, y_vf2, y_vf3 = visualize_principal_curvature(
                                                        Y_verts, 
                                                        # Y_faces, 
                                                        pd1=Y_Curv.PD1, 
                                                        pv1=Y_PV1_norm * avg_y_edge_len, 
                                                        pd2=Y_Curv.PD2,
                                                        pv2=Y_PV2_norm * avg_y_edge_len,
                                                        pd3=Y_Curv.PD3,
                                                        pv3=Y_PV3_norm * avg_y_edge_len,
                                                        scale=0.1)
    
    norm_X_vert = (X_verts - X_verts.min()) / (X_verts.max() - X_verts.min())
    norm_Y_vert = (Y_verts - Y_verts.min()) / (Y_verts.max() - Y_verts.min())
    matching = OT_registration([X_PV2_norm, norm_X_vert], 
                               [Y_PV2_norm, norm_X_vert],
                               "shape0")
    
    polyscope.init()
    ps_cloud = polyscope.register_point_cloud("Mug cup", X_verts)
    ps_cloud.add_scalar_quantity("PV1", X_PV1_norm)
    ps_cloud.add_scalar_quantity("PV2", X_PV2_norm)
    ps_cloud.add_scalar_quantity("PV3", X_PV3_norm)
    polyscope.show()

    # 回転行列の作成.おそらくできていない.
    # 主曲率方向自体はすべて右手座標系になっている.
    # try:
    #     x_rot = create_rotation_matrix(X_Curv.PD1, X_Curv.PD2, X_Curv.PD3)
    #     print(f'created x rotation matricies')
    #     y_rot = create_rotation_matrix(Y_Curv.PD1, Y_Curv.PD2, Y_Curv.PD3)
    #     print(f'created y rotation matricies')
    # except ValueError as e:
    #     print(str(e))
    # coordinate_frame = create_coordinate_frame_mesh(size=0.1)
    # rotated_frames = apply_rotation_matrices(coordinate_frame, x_rot, X_verts)
    # o3d.visualization.draw_geometries([coordinate_frame] + rotated_frames)
    # show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, 
    #                     y_mesh, y_vf1, y_vf2, y_vf3, 
    #                     sync_view=False,
    #                     viz_pca=False)


    raise 
    ################### pricipal curvatureをpcaした結果に基づく初期位置姿勢推定
    ################### 右手座標系と左手座標系が逆になるので要修正
    projected_X_d1, X_pd1_principal_comp = principal_analysis(X_verts, 
                                                            #   X_PV1_norm
                                                              X_Curv.PV1,
                                                              )
    projected_Y_d1, Y_pd1_principal_comp = principal_analysis(Y_verts, 
                                                            #   Y_PV1_norm
                                                              Y_Curv.PV1,
                                                              )

    projected_X_d2, X_pd2_principal_comp = principal_analysis(X_verts, 
                                                            #   X_PV2_norm
                                                                X_Curv.PV2
                                                              )
    projected_Y_d2, Y_pd2_principal_comp = principal_analysis(Y_verts, 
                                                            #   Y_PV2_norm
                                                            Y_Curv.PV2
                                                              )

    projected_X_d3, X_pd3_principal_comp = principal_analysis(X_verts, 
                                                            #   X_PV3_norm
                                                            X_Curv.PV3
                                                              )
    projected_Y_d3, Y_pd3_principal_comp = principal_analysis(Y_verts, 
                                                            #   Y_PV3_norm
                                                            Y_Curv.PV3
                                                              )
    
    # Tangent PCA
    # X_Curv_PD2_6d = np.hstack((X_verts, X_Curv.PD2))
    # space = SpecialEuclidean(n=3, point_type="vector")
    # mean.fit(X_Curv_PD2_6d)
    space = Hypersphere(dim=2)
    # mean = FrechetMean(space)
    # mean.fit(X_Curv.PD3)
    # mean_estimate = mean.estimate_
    tpca = TangentPCA(space, n_components=2)
    tpca.fit(X_Curv.PD1, base_point=X_pd1_principal_comp[0])
    geodesic_pd1_x0 = space.metric.geodesic(initial_point=X_pd1_principal_comp[0], initial_tangent_vec=tpca.components_[0])
    geodesic_pd1_x1 = space.metric.geodesic(initial_point=X_pd1_principal_comp[0], initial_tangent_vec=tpca.components_[1])
    
    tpca.fit(X_Curv.PD1, base_point=X_pd1_principal_comp[1])
    geodesic_pd1_y0 = space.metric.geodesic(initial_point=X_pd1_principal_comp[1], initial_tangent_vec=tpca.components_[0])
    geodesic_pd1_y1 = space.metric.geodesic(initial_point=X_pd1_principal_comp[1], initial_tangent_vec=tpca.components_[1])
    
    tpca.fit(X_Curv.PD1, base_point=X_pd1_principal_comp[2])
    geodesic_pd1_z0 = space.metric.geodesic(initial_point=X_pd1_principal_comp[2], initial_tangent_vec=tpca.components_[0])
    geodesic_pd1_z1 = space.metric.geodesic(initial_point=X_pd1_principal_comp[2], initial_tangent_vec=tpca.components_[1])

    n_steps = 100
    t = np.linspace(-1.0, 1.0, n_steps)

    geodesic_pd1_x0vec = geodesic_pd1_x0(t)
    geodesic_pd1_x1vec = geodesic_pd1_x1(t)
    geodesic_pd1_y0vec = geodesic_pd1_y0(t)
    geodesic_pd1_y1vec = geodesic_pd1_y1(t)
    geodesic_pd1_z0vec = geodesic_pd1_z0(t)
    geodesic_pd1_z1vec = geodesic_pd1_z1(t)
    # このgeodesic上でdeformation fieldとchamfer distanceが最小化になるような点を見つければいいのでは

    # Raw curvature
    x_mesh, x_vf1, x_vf2, x_vf3 = visualize_principal_curvature(
                                                        X_verts, 
                                                        # X_faces, 
                                                        pd1=X_Curv.PD1, 
                                                        pv1=X_PV1_norm * avg_x_edge_len, 
                                                        pd2=X_Curv.PD2,
                                                        pv2=X_PV2_norm * avg_x_edge_len,
                                                        pd3=X_Curv.PD3,
                                                        pv3=X_PV3_norm * avg_x_edge_len,
                                                        scale=0.1)
    y_mesh, y_vf1, y_vf2, y_vf3 = visualize_principal_curvature(
                                                        Y_verts, 
                                                        # Y_faces, 
                                                        pd1=Y_Curv.PD1, 
                                                        pv1=Y_PV1_norm * avg_y_edge_len, 
                                                        pd2=Y_Curv.PD2,
                                                        pv2=Y_PV2_norm * avg_y_edge_len,
                                                        pd3=Y_Curv.PD3,
                                                        pv3=Y_PV3_norm * avg_y_edge_len,
                                                        scale=0.1)
    
    # Principal projected curvature
    # x_mesh, x_vf1, x_vf2, x_vf3 = visualize_principal_curvature(X_verts, 
    #                                                     # X_faces, 
    #                                                     pd1=projected_X_d1, 
    #                                                     pv1=X_Curv.PV1 * avg_x_edge_len, 
    #                                                     pd2=projected_X_d2,
    #                                                     pv2=X_Curv.PV2 * avg_x_edge_len,
    #                                                     pd3=projected_X_d3,
    #                                                     pv3=X_Curv.PV3 * avg_x_edge_len,
    #                                                     scale=5)
    # y_mesh, y_vf1, y_vf2, y_vf3 = visualize_principal_curvature(Y_verts, 
    #                                                     # Y_faces, 
    #                                                     pd1=projected_Y_d1, 
    #                                                     pv1=Y_Curv.PV1 * avg_y_edge_len, 
    #                                                     pd2=projected_Y_d2,
    #                                                     pv2=Y_Curv.PV2 * avg_y_edge_len,
    #                                                     pd3=projected_Y_d3,
    #                                                     pv3=Y_Curv.PV3 * avg_y_edge_len,
    #                                                     scale=5)


    X_mesh_norm = mesh_norm(show_X_verts, show_X_faces)
    Y_mesh_norm = mesh_norm(show_Y_verts, show_Y_faces)
    # X_mesh_norm = mesh_norm(X_verts)
    # Y_mesh_norm = mesh_norm(Y_verts)

    X_sphere = show_distribution_lrf(X_mesh_norm,
                                    X_Curv.PD1, X_Curv.PD2, X_Curv.PD3, 
                                    # X_Curv.PV1, X_Curv.PV2, X_Curv.PV3, 
                                    X_PV1_norm, X_PV2_norm, X_PV3_norm, 
                                    X_pd1_principal_comp, X_pd2_principal_comp,
                                    X_pd3_principal_comp
                                   )

    X_sphere.subplot(0, 0)
    # X_sphere.add_points(geodesic_pd1_x0vec, color='#DE3163', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_x1vec, color='#6495ED', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_x0vec[0,:], color='#FF66FF', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_x0vec[-1,:], color='#FF3300', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_x1vec[0,:], color='#000099', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_x1vec[-1,:], color='#00FFFF', render_points_as_spheres=True, point_size=15)

    # X_sphere.add_points(geodesic_pd1_y0vec, color='#DE3163', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_y1vec, color='#6495ED', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_y0vec[0,:], color='#FF66FF', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_y0vec[-1,:], color='#FF3300', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_y1vec[0,:], color='#000099', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_y1vec[-1,:], color='#00FFFF', render_points_as_spheres=True, point_size=15)

    # X_sphere.add_points(geodesic_pd1_z0vec, color='#DE3163', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_z1vec, color='#6495ED', render_points_as_spheres=True, point_size=10)
    # X_sphere.add_points(geodesic_pd1_z0vec[0,:], color='#FF66FF', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_z0vec[-1,:], color='#FF3300', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_z1vec[0,:], color='#000099', render_points_as_spheres=True, point_size=15)
    # X_sphere.add_points(geodesic_pd1_z1vec[-1,:], color='#00FFFF', render_points_as_spheres=True, point_size=15)
    X_sphere.show()
    Y_sphere = show_distribution_lrf(Y_mesh_norm,
                                    Y_Curv.PD1, Y_Curv.PD2, Y_Curv.PD3, 
                                    # Y_Curv.PV1, Y_Curv.PV2, Y_Curv.PV3, 
                                    Y_PV1_norm, Y_PV2_norm, Y_PV3_norm,
                                    Y_pd1_principal_comp, Y_pd2_principal_comp, 
                                    Y_pd3_principal_comp
                                   )
    Y_sphere.show()

    show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, 
                         y_mesh, y_vf1, y_vf2, y_vf3, 
                         X_pd1_principal_comp,
                         Y_pd1_principal_comp,
                         sync_view=False,
                         viz_pca=False)
    
    # show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, X_pd2_principal_comp,
    #                      y_mesh, y_vf1, y_vf2, y_vf3, Y_pd2_principal_comp,
    #                      sync_view=False)

    # show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, X_pd3_principal_comp,
    #                      y_mesh, y_vf1, y_vf2, y_vf3, Y_pd3_principal_comp,
    #                      sync_view=False)