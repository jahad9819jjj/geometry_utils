import igl
import numpy as np
import pyvista
from pyvista import examples

def pyvista_face_to_triangles(pyvista_face):
    return pyvista_face[1:].reshape(-1, 3)

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

    return mesh, vf1, vf2, vf3

def principal_analysis(data, weights):
    from sklearn.decomposition import PCA
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    weights_column = weights_norm.reshape(-1, 1)
    weighted_data = data * weights_column
    pca = PCA(n_components=3) 
    pca.fit(weighted_data)
    principal_components = pca.components_
    projected_data = pca.transform(weighted_data)
    return projected_data, principal_components
    

if __name__ == "__main__":
    # path = '/Users/jinhirai/Downloads/Codes/3d_deeplearning/stanford-bunny_translatd_decimate036.obj'
    path = '/Users/jinhirai/Downloads/Codes/3d_deeplearning/stanford-bunny_rotated_decimate036.obj'
    verts, triangles = igl.read_triangle_mesh(path)
    d1, d2, v1, v2 = igl.principal_curvature(
                                            verts, 
                                            triangles,
                                            radius=5,
                                            use_k_ring=True)
    v3 = 0.5 * (v1 + v2)
    d3 = np.cross(d1, d2)
    sample_rate  = 0.05
    sample_rate = 1.0
    sample_indices = np.random.choice(len(verts), int(len(verts) * sample_rate), replace=False)
    
    v1_norm = (v1 - v1.min()) / (v1.max() - v1.min())
    v2_norm = (v2 - v2.min()) / (v2.max() - v2.min())
    v3_norm = (v3 - v3.min()) / (v3.max() - v3.min())
    projected_d1, principal_comp = principal_analysis(d1, v1_norm)
    msh, vf1, vf2, vf3 = visualize_principal_curvature(verts, triangles, d1, v1_norm, d2, v2_norm, d3, v3_norm,
                                                       indices=sample_indices,
                                                       scale=0.01
                                                       )
    plotter = pyvista.Plotter()
    origin = np.array([[0, 0, 0]])
    for i, pc in enumerate(principal_comp):
        if i == 0:
            color = 'red'
        elif i == 1:
            color = 'green'
        else:
            color = 'blue'
        
        arrow = pyvista.Arrow(origin, pc, shaft_radius=0.02, tip_radius=0.04, tip_length=0.1, scale=0.1)
        plotter.add_mesh(arrow, color=color)

    plotter.add_mesh(msh, color='gray', opacity=0.3)
    plotter.add_mesh(vf1, line_width=2, color='r')
    plotter.add_mesh(vf2, line_width=2, color='g')
    plotter.add_mesh(vf3, line_width=2, color='b')
    plotter.show()