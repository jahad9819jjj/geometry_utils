import numpy as np
import open3d as o3d
import pyvista as pv
import copy

def is_right_handed(PD1, PD2, PD3, tolerance=1e-6):
    cross_product = np.cross(PD1, PD2)
    return np.allclose(cross_product, PD3, atol=tolerance)

def convert_to_right_handed(PD1, PD2, PD3):
    if not is_right_handed(PD1, PD2, PD3):
        print(f'[WARN] modify to right handed.')
        PD2 = -PD2
    return PD1, PD2, PD3

def is_valid_rotation_matrix(PD1, PD2, PD3, tolerance=1e-5):
    # PD1, PD2, PD3が単位ベクトルかどうかを確認
    if not (np.allclose(np.linalg.norm(PD1, axis=1), 1, atol=tolerance) and
            np.allclose(np.linalg.norm(PD2, axis=1), 1, atol=tolerance) and
            np.allclose(np.linalg.norm(PD3, axis=1), 1, atol=tolerance)):
        print(f'[Error] principal directions are not unit vectors.')
        return False
    
    # PD1, PD2, PD3が互いに直交しているかどうかを確認
    if not (np.allclose(np.sum(PD1 * PD2, axis=1), 0, atol=tolerance) and
            np.allclose(np.sum(PD2 * PD3, axis=1), 0, atol=tolerance) and
            np.allclose(np.sum(PD3 * PD1, axis=1), 0, atol=tolerance)):
        print(f'[Error] principal directions are orthogonal.')
        return False
    
    return True

def create_rotation_matrix(PD1, PD2, PD3):
    PD1, PD2, PD3 = convert_to_right_handed(PD1, PD2, PD3)
    
    if is_valid_rotation_matrix(PD1, PD2, PD3):
        rotation_matrix = np.stack((PD1, PD2, PD3), axis=1)
        return rotation_matrix
    else:
        raise ValueError("Invalid input vectors. Cannot create rotation matrix.")

def create_coordinate_frame_mesh(size=1.0, origin=[0, 0, 0]):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return frame

def apply_rotation_matrices(coordinate_frame, rotation_matrices, translation_vecs):
    rotated_frames = []
    for idx, matrix in enumerate(rotation_matrices):
        crd = copy.copy(coordinate_frame)
        crd = crd.rotate(matrix)
        crd = crd.translate(translation_vecs[idx])
        rotated_frames.append(crd)
    return rotated_frames
