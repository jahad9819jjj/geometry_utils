import numpy as np
import pyvista
import open3d as o3d
import time

from kde_gpu import (
    pdf_keops, plot_density, plot_curvature_density_3d, 
    plot_curvature_density_keops_3d,
    plot_curvature_density_keops_3d_pyvista_multiple
)
import matplotlib.pyplot as plt

from spatial_graph import SpatialGraph

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from geomstats.geometry.special_euclidean import SpecialEuclidean
import geomstats.backend as gs
import geomstats.visualization as visualization

import copy

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

    return verts, faces, curv_info

def compute_average_edge_length(vertices, faces):
    """
    vertices: numpy array of shape (num_vertices, 3)
    faces: numpy array of shape (num_faces, 3)
    """
    # 各面の頂点座標を取得
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # エッジの長さを計算
    edge_lengths = np.concatenate([
        np.linalg.norm(v1 - v2, axis=1),
        np.linalg.norm(v2 - v3, axis=1),
        np.linalg.norm(v3 - v1, axis=1)
    ])
    
    # 平均エッジ長さを計算
    average_edge_length = np.mean(edge_lengths)
    
    return average_edge_length


def get_pyvista_coord_frame(size=10):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size)
    verts = np.asarray(frame.vertices)
    triangles = np.asarray(frame.triangles)
    faces = np.concatenate((np.full((triangles.shape[0], 1), 3), triangles), axis=1)
    frame_pv = pyvista.PolyData(verts, faces)
    frame_pv.point_data["color"] = np.asarray(frame.vertex_colors)
    return frame_pv

def visualize_principal_curvature(verts, faces, pd1, pv1, pd2, pv2, pd3, pv3, indices=None, scale=None):
    ''' 
    sample_indices = np.random.choice(mesh.n_points, int(mesh.n_points * sample_rate), replace=False)
    visualize_principal_curvature(..., indices=sample_indices)
    '''
    triangles = np.concatenate((np.full((faces.shape[0], 1), 3), faces), axis=1)
    mesh = pyvista.PolyData(verts, triangles)
    pcd = pyvista.PolyData(verts)
    pcd.point_data["pd1"] = pd1
    pcd.point_data["pv1"] = pv1
    pcd.point_data["pd2"] = pd2
    pcd.point_data["pv2"] = pv2
    pcd.point_data["normals"] = pd3
    pcd.point_data["normals_scale"] = pv3

    if indices is None:
        indices = np.arange(mesh.n_points)

    sampled_pcd = pcd.extract_points(indices)

    # 主曲率方向ベクトルを表示するためのラインを作成
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

    # plotter.add_mesh(mesh, color='gray', opacity=0.3)
    # plotter.add_mesh(vf1, 
    #                 #  color='blue', 
    #                  line_width=2, 
    #                  cmap='jet'
    #                  )
    # plotter.add_mesh(vf2, 
    #                 #  color='green', 
    #                  line_width=2,
    #                  cmap='jet')
    # plotter.show()

    return mesh, vf1, vf2, vf3

def check_rotation_matrices(rots):
    for idx, rot in enumerate(rots):
        check_I = rot @ np.linalg.inv(rot)
        check_I = np.allclose(check_I, np.eye(3))
        print(f"{idx}'s Identity = {check_I}")
        if not check_I:
            raise ValueError(f'Rotation matrix is not identity. {rot}')

def align_knn(  verts, 
                faces, 
                pd1, 
                pv1, 
                pd2,
                pv2,
                pd3,
                pv3,
                scale=10):
    # なかなかそろえるのは厳しそう(反転：非反転の割合が半分近くなので)
    # TODO: igl.principal_curvature()の実装を見直し, Lie代数を適用できるならばcurvatureの方で矯正させる
    pv_faces = np.concatenate((np.full((faces.shape[0], 1), 3), faces), axis=1)
    mesh = pyvista.PolyData(verts, pv_faces)

    start = time.time()
    sg = SpatialGraph(verts, faces)
    print(f'Elapsed time : {time.time() - start}')
    
    q = verts[0]
    start = time.time()
    nearest_idxs, nearest_neighbors, nearest_distances = sg.k_nearest_neighbors(q, k=100)
    print(f'Elapsed time : {time.time() - start}')

    nn_vert = verts[nearest_idxs]
    nn_pd1 = pd1[nearest_idxs]
    nn_pd2 = pd2[nearest_idxs]
    nn_pd3 = pd3[nearest_idxs]
    nn_pv1 = pv1[nearest_idxs]
    nn_pv2 = pv2[nearest_idxs]
    nn_pv3 = pv3[nearest_idxs]

    # TODO: visualize_principal_curvatureを用いて, NNのLRF分布をみる.
    # TODO: 適切にLRFをそろえる
    # Gram–Schmidtでも可?
    # LRFにおける空間はHaussdorff spaceというらしい, http://cat.phys.s.u-tokyo.ac.jp/lecture/MP3_16/maph3.pdf
    se3 = SpecialEuclidean(n=3, point_type="matrix")
    rotation_matrices = np.array([
        np.vstack((nn_pd1[i], nn_pd2[i], nn_pd3[i])) for i in range(len(nn_pd1))
        ])
    check_rotation_matrices(rotation_matrices)

    lrfs_euc = np.zeros((100, 4, 4))
    lrfs_euc[:, :3, :3] = rotation_matrices
    lrfs_euc[:, :3, 3] = nn_vert 
    lrfs_euc[:, 3, 3] = 0

    lrfs_se3 = np.array([se3.log(lrf_euc, base_point=se3.identity) for lrf_euc in lrfs_euc])
    mean_lrf_se3 = np.mean(lrfs_se3, axis=0)
    mean_lrf_euc = se3.exp(mean_lrf_se3)
    
    mean_mesh = get_pyvista_coord_frame()
    mean_lrf_mesh = copy.copy(mean_mesh).transform(mean_lrf_euc)
    
    diff_mat_se3 = lrfs_se3 - mean_lrf_se3
    diff_mat_euc = se3.exp(diff_mat_se3)
    diff_lrf_mesh = [copy.copy(mean_mesh).transform(diff_mat) for diff_mat in diff_mat_euc]


    nn_mesh, vf1, vf2, vf3 = visualize_principal_curvature(verts, faces, pd1, pv1, pd2, pv2, pd3, pv3, nearest_idxs)
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh, color='gray', opacity=0.3)
    plotter.add_mesh(vf1, color='blue', line_width=2)
    plotter.add_mesh(vf2, color='green', line_width=2)
    plotter.add_mesh(vf3, color='red', line_width=2)
    plotter.add_mesh(mean_mesh, rgb=True)
    plotter.add_mesh(mean_lrf_mesh, rgb=True)
    for diff_mesh in diff_lrf_mesh:
        plotter.add_mesh(diff_mesh, rgb=True)
    plotter.show()
    raise

    # debug
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(mesh, color="white", show_edges=True, opacity=0.5)
    # plotter.add_points(q, color="red", point_size=20)
    # plotter.add_points(nearest_neighbors, color="blue", point_size=20)
    # for i in range(len(nearest_neighbors)):
    #     line = pyvista.Line(q, nearest_neighbors[i])
    #     plotter.add_mesh(line, color="green", line_width=2)
    # plotter.show()


def show_two_mesh_and_vf(mesh_x, x_vf1, x_vf2, x_vf3, mesh_y, y_vf1, y_vf2, y_vf3):
    # Create a subplot with 2 rows and 1 column
    plotter = pyvista.Plotter(shape=(1, 2))

    # Add the first mesh and vector fields to the first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_x, color='gray', opacity=0.3)
    plotter.add_mesh(x_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(x_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(x_vf3, line_width=2, color='b')
    plotter.add_title("Mesh X")

    # Add the second mesh and vector fields to the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_y, color='gray', opacity=0.3)
    plotter.add_mesh(y_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(y_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(y_vf3, line_width=2, color='b')
    plotter.add_title("Mesh Y")

    # Link the camera between subplots
    plotter.link_views()

    # Show the plot
    plotter.show()



if __name__ == "__main__":
    # 主曲率方向, 主曲率値が格納されている(この例ではlibigl.principal_curvature()を用いた値を格納している)
    X_verts, X_faces, X_Curv = load_ply_mesh("/mnt/d/1_autoannotate/bcpd/debug/curv/X_mesh.ply")
    Y_verts, Y_faces, Y_Curv = load_ply_mesh("/mnt/d/1_autoannotate/bcpd/debug/curv/Y_mesh.ply")

    avg_x_edge_len = compute_average_edge_length(X_verts, X_faces)
    avg_y_edge_len = compute_average_edge_length(Y_verts, Y_faces)
    avg_x_edge_len = 1
    avg_y_edge_len = 1

    X_Curv.PD3 = np.cross(X_Curv.PD1, X_Curv.PD2)
    X_Curv.PV3 = (X_Curv.PV1 + X_Curv.PV2) / 2
    Y_Curv.PD3 = np.cross(Y_Curv.PD1, Y_Curv.PD2)
    Y_Curv.PV3 = (Y_Curv.PV1 + Y_Curv.PV2) / 2

    align_knn(X_verts, X_faces,
                X_Curv.PD1, 
                X_Curv.PV1 * avg_x_edge_len, 
                X_Curv.PD2,
                X_Curv.PV2 * avg_x_edge_len,
                X_Curv.PD3,
                X_Curv.PV3 * avg_x_edge_len,
                )

    x_mesh, x_vf1, x_vf2, x_vf3 = visualize_principal_curvature(X_verts, 
                                                        X_faces, 
                                                        X_Curv.PD1, 
                                                        X_Curv.PV1 * avg_x_edge_len, 
                                                        X_Curv.PD2,
                                                        X_Curv.PV2 * avg_x_edge_len,
                                                        X_Curv.PD3,
                                                        X_Curv.PV3 * avg_x_edge_len,
                                                        scale=10)
    y_mesh, y_vf1, y_vf2, y_vf3 = visualize_principal_curvature(Y_verts, 
                                                        Y_faces, 
                                                        Y_Curv.PD1, 
                                                        Y_Curv.PV1 * avg_y_edge_len, 
                                                        Y_Curv.PD2,
                                                        Y_Curv.PV2 * avg_y_edge_len,
                                                        Y_Curv.PD3,
                                                        Y_Curv.PV3 * avg_y_edge_len,
                                                        scale=10)
    
    show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3,
                         y_mesh, y_vf1, y_vf2, y_vf3)
    
    # norm_curv_x1 = ( X_Curv.PV1 - X_Curv.PV1.min()) / (X_Curv.PV1.max() - X_Curv.PV1.min())
    # norm_curv_x2 = ( X_Curv.PV2 - X_Curv.PV2.min()) / (X_Curv.PV2.max() - X_Curv.PV2.min())

    # density_x1 = plot_curvature_density_keops_3d(X_verts, X_Curv.PV1)
    plot_curvature_density_keops_3d_pyvista_multiple(X_verts, Y_verts, 
                                                     X_Curv.PV1, X_Curv.PV2,
                                                     Y_Curv.PV1, Y_Curv.PV2)


    # norm_curv_y1 = ( Y_Curv.PV1 - Y_Curv.PV1.min()) / (Y_Curv.PV1.max() - Y_Curv.PV1.min())
    # norm_curv_y2 = ( Y_Curv.PV2 - Y_Curv.PV2.min()) / (Y_Curv.PV2.max() - Y_Curv.PV2.min())
    norm_curv_y1 = Y_Curv.PV1 
    norm_curv_y2 = Y_Curv.PV2 
