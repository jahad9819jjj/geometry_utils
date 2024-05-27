# robust_laplace pcdを適用
# pyFMでrobust_laplaceのeigen vecを入力し, 対応を推定
import robust_laplacian
import numpy as np
import polyscope
import scipy.sparse.linalg as sla
import open3d as o3d

if __name__ == "__main__":
    src_path = '/Users/jinhirai/Downloads/Dataset/Mug/Mug4_remesh.ply' #annotation
    trg_path = '/Users/jinhirai/Downloads/Dataset/Mug/Mug10_remesh.ply' #target

    src_mesh = o3d.io.read_triangle_mesh(src_path)
    trg_mesh = o3d.io.read_triangle_mesh(trg_path)

    src_verts = np.asarray(src_mesh.vertices); src_colors = np.asarray(src_mesh.vertex_colors); src_triangles = np.asarray(src_mesh.triangles)
    trg_verts = np.asarray(trg_mesh.vertices); trg_colors = np.asarray(trg_mesh.vertex_colors); trg_triangles = np.asarray(trg_mesh.triangles)

    L_src, M_src = robust_laplacian.point_cloud_laplacian(src_verts)
    L_trg, M_trg = robust_laplacian.point_cloud_laplacian(trg_verts)

    # (or for a mesh)
    # L, M = robust_laplacian.mesh_laplacian(verts, faces)

    # Compute some eigenvectors
    n_eig = 10
    evals_src, evecs_src = sla.eigsh(L_src, n_eig, M_src, sigma=1e-8)
    evals_trg, evecs_trg = sla.eigsh(L_trg, n_eig, M_trg, sigma=1e-8)

    polyscope.init()

    polyscope_src_cloud = polyscope.register_point_cloud("Source Cloud", src_verts)
    
    offset = 0.5
    trg_verts[:, 0] -= offset
    polyscope_trg_cloud = polyscope.register_point_cloud("Target Cloud", trg_verts)

    for i in range(n_eig):
        polyscope_src_cloud.add_scalar_quantity(f"Source Eigenvector_{i}", evecs_src[:,i], enabled=True)
        polyscope_trg_cloud.add_scalar_quantity(f"Target Eigenvector_{i}", evecs_trg[:,i], enabled=True)

    polyscope.show()