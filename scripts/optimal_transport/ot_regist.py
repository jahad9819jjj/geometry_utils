import numpy as np
import open3d as o3d
import torch
import os
import sys
from geomloss import SamplesLoss
from pykeops.torch import LazyTensor
from plyfile import PlyData, PlyElement
import pyvista
import time
import matplotlib.pyplot as plt
sys.path.append('/workspace/robust_opt')
import pointnet2.lib.pointnet2_utils as pointnet2_utils
from lib.utils import get_spherical_pcd, get_pcd_from_spcd, torchcu2numpy
import pygeodesic.geodesic as geod

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.grassmannian import Grassmannian

# Check Gromov wasserstein
# import jax
# import jax.numpy as jnp
# from ott.geometry import pointcloud, costs
# from ott.problems.quadratic import quadratic_problem
# from ott.solvers.quadratic import gromov_wasserstein
# from ott.tools import plot
from lib.utils import optimize, compute_fpfh, get_o3d_pcd, ransac_fpfh

def sampling_farthest_points(points, colors, npoint=10000):
    """ Sampling by farthest.

    Args:
        points (np.array): NxD
        npoint (int): M

    Returns:
        np.array : MxD
    """
    np.random.seed(0)
    N, D = points.shape
    xyz = points[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    colors = colors[centroids.astype(np.int32)]
    return points, colors

def create_sphere(n_samples=1000):
    """Creates a uniform sample on the unit sphere."""
    n_samples = int(n_samples)
    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    points = np.vstack((x, y, z)).T
    weights = np.ones(n_samples) / n_samples
    return tensor(weights), tensor(points)

def to_measure(points, colors=None, triangles=None, normals=None):
    """Turns a triangle into a weighted point cloud."""
    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces
    print(
        "File loaded, and encoded as the weighted sum of {:,} atoms in 3D.".format(
            len(X)
        )
    )
    if colors is not None:
        # clr_A, clr_B, clr_C = colors[triangles[:, 0]], colors[triangles[:, 1]], colors[triangles[:, 2]]
        # clr_X = (clr_A + clr_B + clr_C) / 3
        clr_X = colors[triangles[:, 0]]
    if normals is not None:
        nrm_X = normals[triangles[:, 0]]
    if colors is not None:
        return tensor(S / np.sum(S)), tensor(X), tensor(clr_X), tensor(nrm_X)  
    else:
        return tensor(S / np.sum(S)), tensor(X), tensor(nrm_X) 

def load_ply_file(fname):
    """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""
    # Load the data, and read the connectivity information:
    plydata = PlyData.read(fname)
    triangles = np.vstack(plydata["face"].data["vertex_indices"])
    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]])

    return to_measure(points, triangles)

def normalize(measure, n=None, with_color:bool=False):
    """Reduce a point cloud to at most n points and normalize the weights and point cloud."""
    if len(measure) == 2:
        weights, locations = measure
    elif len(measure) == 3 and (with_color):
        weights, locations, colors = measure
    elif len(measure) == 3 and not (with_color):
        weights, locations, normals = measure
    elif len(measure) == 4 and (with_color):
        weights, locations, normals, colors = measure
    else:
        raise ValueError(f'Error')
    N = len(weights)
    if n is not None and n < N:
        n = int(n)
        indices = torch.randperm(N)
        indices = indices[:n]
        if with_color:
            weights, locations, colors, normals = weights[indices], locations[indices], colors[indices], normals[indices]
        else:
            weights, locations, normals = weights[indices], locations[indices], normals[indices]
    weights = weights / weights.sum()
    if with_color:
        weights, locations, colors, normals = weights.contiguous(), locations.contiguous(), colors.contiguous(), normals.contiguous()
    else:
        weights, locations, normals = weights.contiguous(), locations.contiguous(), normals.contiguous()

    # Center, normalize the point cloud
    mean = (weights.view(-1, 1) * locations).sum(dim=0)
    locations -= mean
    std = (weights.view(-1) * (locations ** 2).sum(dim=1).view(-1)).sum().sqrt()
    locations /= std
    if with_color:
        return weights, locations, colors, normals
    else:
        return weights, locations, normals

def create_mesh_from_pcd(pcd, method:str='alpha_shape', params:dict={}):
    mesh = None
    if method == "alpha_shape":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 
                                                                             alpha=params['alpha'])
    elif method == "ball_pivot":
        pcd.estimate_normals()
        radius = o3d.utility.DoubleVector(params['radii'])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radius)
    elif method == "poisson":
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,
                                                                         params['depth'],
                                                                         params['width'],
                                                                         params['scale'],
                                                                         params['linear_fit'],
                                                                         params['n_threads']
                                                                         )
    else:
        raise ValueError(f'No such that method : {method}')
    return mesh

def ot_sampling(
        pcd:o3d.geometry.PointCloud,
        num=1000,
        blur=0.02,
        sparse=False
):
    input_pts = np.asarray(pcd.points)
    input_clrs = np.asarray(pcd.colors)
    input_pts_torch = torch.Tensor(input_pts).to('cuda')
    input_pts_torch = input_pts_torch.unsqueeze(0)
    input_clrs_torch = torch.Tensor(input_clrs).to('cuda')
    input_clrs_torch = input_clrs_torch.unsqueeze(0)
    input, sampled_pts, sampled_colors = wasserstein_barycenter_mapping(input_pts_torch,
                                                    input_clrs_torch,
                                                    num,
                                                    blur,
                                                    sparse
                                                    )
    sampled_pts = sampled_pts.squeeze()
    sampled_colors = sampled_colors.squeeze()
    cpu_sampled_pts = sampled_pts.detach().cpu().numpy()
    cpu_sampled_clrs = sampled_colors.detach().cpu().numpy()
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(cpu_sampled_pts)
    sampled_pcd.colors = o3d.utility.Vector3dVector(cpu_sampled_clrs)
    return sampled_pcd

def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    
    # https://github.com/jahad9819jjj/robust_opt/blob/main/pointnet2/lib/src/group_points_gpu.cu
    # https://github.com/jahad9819jjj/robust_opt/blob/fa910327d03a850981cf95fac4e5803f727437d1/pointnet2/lib/pointnet2_utils.py#L229
    new_points = pointnet2_utils.grouping_operation(
        points_flipped, knn_idx.int()
    ).permute(0, 2, 3, 1)

    return new_points
    
def wasserstein_barycenter_mapping(input_points, 
                                   input_colors,
                                   wsm_num,blur=0.01,
                                   sparse=False):
    """

    :param input_points: BxNxD
    :param wsm_num: M
    :param blur:
    :return:
    """
    def NN():
        def compute(pc1, pc2, colors2):
            pc_i = LazyTensor(pc1[:,:,None])
            pc_j = LazyTensor(pc2[:,None])
            dist2 = pc_i.sqdist(pc_j)
            K_min, index = dist2.min_argmin(dim=2)
            Kmin_pc3 = index_points_group(pc2,index)
            sampled_colors = index_points_group(colors2, index)
            return Kmin_pc3, sampled_colors
        return compute
    grad_enable_record =torch.is_grad_enabled()
    geomloss  = SamplesLoss(loss='sinkhorn',blur=blur, scaling=0.8,reach=None,debias=False,potentials=True,backend='multiscale',truncate=5 if sparse else None)
    points2 = input_points
    B, n_input, device = input_points.shape[0], input_points.shape[1], input_points.device
    weights2 = torch.ones(B, n_input).to(device)/n_input
    prob = torch.ones(n_input).to(device)
    idx = prob.multinomial(num_samples=wsm_num, replacement=False)
    points1 = points2[:,idx].contiguous().clone()
    weights1 = torch.ones(B, wsm_num).to(device)/wsm_num
    device = points2.device
    sqrt_const2 = torch.tensor(np.sqrt(2),dtype=torch.float32, device=device)
    F_i, G_j = geomloss(weights1, points1, weights2,  points2)
    B, N, M, D = points1.shape[0], points1.shape[1], points2.shape[1], points2.shape[2]
    torch.set_grad_enabled(grad_enable_record)
    a_i, x_i = LazyTensor(weights1.view(B,N, 1, 1)), LazyTensor(points1.view(B,N, 1, -1))
    b_j, y_j = LazyTensor(weights2.view(B,1, M, 1)), LazyTensor(points2.view(B,1, M, -1))
    F_i, G_j = LazyTensor(F_i.view(B,N, 1, 1)), LazyTensor(G_j.view(B,1, M, 1))
    xx_i = x_i / (sqrt_const2 * blur)
    yy_j = y_j / (sqrt_const2 * blur)
    f_i = a_i.log() + F_i / blur ** 2
    g_j = b_j.log() + G_j/ blur ** 2
    C_ij = ((xx_i - yy_j) ** 2).sum(-1)
    log_P_ij = (f_i + g_j - C_ij)
    position_to_map = LazyTensor(points2.view(B,1, M, -1))  # Bx1xMxD
    mapped_position = log_P_ij.sumsoftmaxweight(position_to_map,dim=2)
    nn_interp = NN()
    sampled_points, sampled_colors = nn_interp(mapped_position, points2, input_colors)
    return points1,sampled_points, sampled_colors

def av_dist(array1, array2):
    """
    arguments:
        array1, array2: both size: (num_points, num_feature)
    returns:
        distances: size: (1,)
    """
    distances = np.sum(np.square(np.expand_dims(array1, axis=1) - np.expand_dims(array2, axis=0)), axis=-1)
    distances = np.min(distances, axis=1)
    distances = np.mean(distances)
    return distances

def chamfer_distance(pointsA, pointsB):
    """Chamfer Distance評価指標

    Args:
        pointsA (np.array): 点群情報A
        pointsB (np.array): 点群情報B

    Returns:
        float: CD値
    """
    dist_A = av_dist(pointsA, pointsB)
    dist_B = av_dist(pointsB, pointsA) 
    return dist_A + dist_B


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    numpy = lambda x: x.detach().cpu().numpy()
    
    plotter = pyvista.Plotter()

    # annotation_pcd_path = '/workspace/Data/NEDO_3D-Part-Function_Dataset/Annotations/Bowl/Bowl1.ply'
    annotation_pcd_path = '/workspace/Data/Test/mug28_adj.ply' # reconstructed with using SSD
    target_pcd_path = '/workspace/Data/Test/mug17_adj.ply' # reconstructed with using SSD
    # annotation_pcd_path = '/workspace/Data/Test/mug28_sample_f.ply' # reconstructed with using SSD
    # target_pcd_path = '/workspace/Data/Test/mug17_sample_f.ply' # reconstructed with using SSD
    def load_ply(path, with_color=True):
        plydata = PlyData.read(path)
        triangles = np.vstack(plydata["face"].data["vertex_indices"])
        points = np.vstack(
            [
                [v['x'], v['y'], v['z'] ]
            ] for v in plydata["vertex"]
        )
        normals = np.vstack(
                [
                    [v['nx'], v['ny'], v['nz'] ]
                ] for v in plydata['vertex']
            )
        if with_color:
            colors = np.vstack(
                [
                    [v['red'], v['green'], v['blue'], v['alpha']]
                ] for v in plydata["vertex"]
            )
            weights, pcd_tensor, clr_tensor, nrm_tensor = to_measure(points, colors, triangles, normals)
            return weights, pcd_tensor, triangles, nrm_tensor,  clr_tensor
        else:
            weights, pcd_tensor, nrm_tensor = to_measure(points, colors=None, triangles=triangles, normals=normals)
            return weights, pcd_tensor, triangles, nrm_tensor

    ann_w, ann_pcd, ann_face, ann_nrm, ann_clr = load_ply(annotation_pcd_path, with_color=True)
    trg_w, trg_pcd, trg_face, trg_nrm = load_ply(target_pcd_path, with_color=False)

    # def create_pyvista_polydata(points, triangles):
    #     faces = np.hstack([[3, *tri] for tri in triangles])  # 各三角形の前に頂点の数(3)を追加
    #     return pyvista.PolyData(points, faces)

    # ann_pts = ann_pcd.detach().cpu().numpy()
    # ann_poly = create_pyvista_polydata(ann_pts, ann_face)
    # trg_idx = ann_poly.find_closest_point(ann_pts[0])
    # src_dists = []
    # for src_pt in ann_pts:
    #     try:
    #         dist = ann_poly.geodesic_distance(ann_poly.find_closest_point(src_pt), 
    #                                         trg_idx)
    #     except:
    #         dist = 5000
    #     src_dists.append(dist)
    # src_dists = np.array(src_dists)
    # plotter.add_points(
    #     ann_pts,
    #     scalars=src_dists,
    #     render_points_as_spheres=True,
    #     point_size=10,
    #     show_scalar_bar=True
    # )
    
    num_pts = 1e4
    norm_ann_w, norm_ann_pts, norm_ann_clrs, norm_ann_nrms = normalize(
                                [ann_w, ann_pcd, ann_nrm, ann_clr],
                                n=1e4,
                                with_color=True
                                )
    norm_trg_w, norm_trg_pts, norm_trg_nrms = normalize(
                                [trg_w, trg_pcd, trg_nrm],
                                n=1e4,
                                with_color=False
                                )
    
    radi = 0.15
    norm_ann_pcd = get_o3d_pcd(norm_ann_pts, norm_ann_clrs, norm_ann_nrms)
    _, norm_ann_fpfh = compute_fpfh(norm_ann_pcd, radius=radi)
    norm_trg_pcd = get_o3d_pcd(norm_trg_pts, nrm=norm_trg_nrms)
    _, norm_trg_fpfh = compute_fpfh(norm_trg_pcd, radius=radi)
    # res = ransac_fpfh(norm_ann_pcd, norm_trg_pcd,
    #                   norm_ann_fpfh, norm_trg_fpfh,
    #                   size=0.1)
    # norm_ann_pcd_transformed = norm_ann_pcd.transform(res.transformation)

    def KLD(a, b, bins=10, epsilon=.00001):

        a_hist, _ = np.histogram(a, bins=bins) 
        b_hist, _ = np.histogram(b, bins=bins)
        

        a_hist = (a_hist+epsilon)/np.sum(a_hist)
        b_hist = (b_hist+epsilon)/np.sum(b_hist)
        
        return np.sum([ai * np.log(ai / bi) for ai, bi in zip(a_hist, b_hist)])
    
    _Ps = torch.Tensor(norm_ann_fpfh.data)
    _Qs = torch.Tensor(norm_trg_fpfh.data)
    Ps = torch.transpose(_Ps, 0, 1).numpy()
    Qs = torch.transpose(_Qs, 0, 1).numpy()
    P = Ps[0]

    sims = []
    for Q in Qs:
        sim = KLD(P, Q, bins=33)
        sims.append(sim)
    sims = np.asarray(sims)

    norm_ann_pts_np = torchcu2numpy(norm_ann_pts)
    norm_ann_clrs_np = torchcu2numpy(norm_ann_clrs)
    plotter.add_points(
        norm_ann_pts_np, 
        scalars=norm_ann_clrs_np[:, :3],
        render_points_as_spheres=True,
        point_size=10,
        rgb=True
    )
    plotter.add_points(
        np.array([norm_ann_pts_np[0]]),
        scalars=np.array([[255, 0, 0]]),
        render_points_as_spheres=True,
        point_size=30,
        rgb=True
    )

    norm_trg_pts_np = torchcu2numpy(norm_trg_pts)
    pplotter = pyvista.Plotter()
    pplotter.add_points(
        norm_trg_pts_np,
        scalars=sims,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=True
    )
    plotter.show(); pplotter.show()
    
    # geomstats
    # sphere = Hypersphere(dim=3, equip=False)
    # norm_ann_fpfh_np = np.asarray(norm_ann_fpfh.data)
    # norm_ann_fpfh_np = norm_ann_fpfh_np.reshape((norm_ann_fpfh_np.shape[1], norm_ann_fpfh_np.shape[0]))
    # projections = sphere.projection(norm_ann_fpfh_np)
    # from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
    # import geomstats.backend as gs
    # geom_sur = DiscreteSurfaces(ann_face)
    # face_areas = geom_sur.face_areas(gs.array(torchcu2numpy(ann_pcd)))


    Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.5, truncate=1)
    def OT_registration(source, target, name):
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

    src_w = norm_ann_w.detach().cpu().numpy()
    src_pts = norm_ann_pts.detach().cpu().numpy()
    trg_w = norm_trg_w.detach().cpu().numpy()
    trg_pts = norm_trg_pts.detach().cpu().numpy()

    # # ott-jax random seed
    # rng = jax.random.PRNGKey(7)
    # rng, *subrngs = jax.random.split(rng, 4)
    # G_xx = pointcloud.PointCloud(x=src_pts, y=src_pts)
    # G_yy = pointcloud.PointCloud(x=trg_pts, y=trg_pts)
    # optimal_transports = optimize(jnp.array(src_w),
    #                               jnp.array(src_pts),
    #                               jnp.array(trg_w),
    #                               jnp.array(trg_pts),
    #                               num_iter=2,
    #                               epsilon=1e-1,
    #                               cost_fn=costs.SqPNorm(p=2))
    # res = optimal_transports[-1]
    # potentials = res.to_dual_potentials()
    # res_pts = potentials.transport(jnp.array(src_pts))
    # res_pcd = o3d.geometry.PointCloud()
    # res_pcd.points = o3d.utility.Vector3dVector(res_pts)
    # res_pcd.colors = o3d.utility.Vector3dVector(
    #     norm_ann_clrs[:, :3].detach().cpu().numpy()
    # )
    # o3d.io.write_point_cloud('./ott_jax_pcd.ply', res_pcd)

    norm_ann_pcd = o3d.geometry.PointCloud()
    norm_ann_pcd.points = o3d.utility.Vector3dVector(
        norm_ann_pts.detach().cpu().numpy()
    )
    norm_ann_pcd.colors = o3d.utility.Vector3dVector(
        norm_ann_clrs[:, :3].detach().cpu().numpy()
    )

    norm_trg_pcd = o3d.geometry.PointCloud()
    norm_trg_pcd.points = o3d.utility.Vector3dVector(
        norm_trg_pts.detach().cpu().numpy()
    )
    norm_trg_pcd.paint_uniform_color([0, 0, 1])

    # Optimal transport
    # norm_ann_spts_np = get_spherical_pcd(torchcu2numpy(norm_ann_pts))
    # norm_trg_spts_np = get_spherical_pcd(torchcu2numpy(norm_trg_pts))
    # norm_ann_spts_tensor = tensor(norm_ann_spts_np)
    # norm_trg_spts_tensor = tensor(norm_trg_spts_np)
    # norm_ann_pts_np = np.asarray(norm_ann_pcd_transformed.points)
    # norm_ann_clrs_np = np.asarray(norm_ann_pcd_transformed.colors)
    norm_ann_pts = tensor(norm_ann_pts_np)
    matching = OT_registration([norm_ann_w, norm_ann_pts],
                               [norm_trg_w, norm_trg_pts],
                               "shape_0")
    # matching = OT_registration([norm_ann_w, norm_ann_spts_tensor],
    #                            [norm_trg_w, norm_trg_spts_tensor],
    #                            "shape_0")
    # matching_spts = matching.detach().cpu().numpy()
    # matching_pts = get_pcd_from_spcd(matching_spts)

    matching_pts = matching.detach().cpu().numpy()
    ot_pcd = o3d.geometry.PointCloud()
    ot_pcd.points = o3d.utility.Vector3dVector(
        matching_pts
    )
    ot_pcd.colors = o3d.utility.Vector3dVector(
        norm_ann_clrs[:, :3].detach().cpu().numpy()
    )
    
    # from lib.kernel import gaussian_kernel
    # G_YY = gaussian_kernel(norm_ann_pts[0], norm_ann_pts, blur=0.5)
    # G_XY = gaussian_kernel(norm_ann_pts, norm_trg_pts, blur=0.01)


    # Visualization self gaussian kernel
    # plotter.add_points(
    #     norm_ann_pts_np,
    #     scalars=G_XY.detach().cpu().numpy(),
    #     render_points_as_spheres=True,
    #     point_size=10,
    #     show_scalar_bar=True
    # )

    # plotter.add_points(
    #     norm_ann_pts_np,
    #     scalars=G_YY.detach().cpu().numpy(),
    #     render_points_as_spheres=True,
    #     point_size=10,
    #     show_scalar_bar=True
    # )

    # plotter.add_points(
    #     norm_ann_pts_np[0],
    #     scalars=np.array([[1, 0, 0, 0.5]]),
    #     render_points_as_spheres=True,
    #     point_size=30,
    #     rgba=True
    # )
    

    plotter.add_points(
        norm_ann_pts_np,
        scalars=norm_ann_clrs_np,
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=False,
        rgba=True
    )

    norm_trg_pts_np = norm_trg_pts.detach().cpu().numpy()
    plotter.add_points(
        norm_trg_pts_np,
        scalars=np.array([[0, 0, 1, 1] for _ in norm_trg_pts]),
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=False,
        rgba=True
    )
    plotter.add_points(
        matching.detach().cpu().numpy(),
        scalars=norm_ann_clrs.detach().cpu().numpy(),
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=False,
        rgba=True
    )

    source_target_dist = chamfer_distance(norm_ann_pts.detach().cpu().numpy(),
                                          norm_trg_pts_np)
    matching_target_dist = chamfer_distance(matching.detach().cpu().numpy(),
                                            norm_trg_pts_np)
    print(f'Before deformation : {source_target_dist}')
    print(f'After deformation : {matching_target_dist}')

    o3d.io.write_point_cloud('./ot_regist.ply', ot_pcd)
    o3d.io.write_point_cloud('./src.ply', norm_ann_pcd)
    o3d.io.write_point_cloud('./trg.ply', norm_trg_pcd)


    raise

    # annotation_pcd = o3d.io.read_point_cloud(annotation_pcd_path)
    # annotation_pcd_ds = annotation_pcd.farthest_point_down_sample(num_samples=10000)
    # annotation_pcd_ds = ot_sampling(annotation_pcd, 5000)
    # annotation_mesh, convex_hull_corr_pcd = annotation_pcd.compute_convex_hull(joggle_inputs=False)
    # annotation_vertices = np.asarray(annotation_mesh.vertices)

    # annotation_mesh_ds = create_mesh_from_pcd(annotation_pcd_ds, 
    #                                           'ball_pivot', 
    #                                           {'radii':[40., 50, 60, 70]})
    
    # target = load_ply_file(annotation_pcd_path)
    # spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=1.0).translate(vert).paint_uniform_color([1, 0, 0]) for vert in annotation_vertices]
    # o3d.visualization.draw_geometries([annotation_mesh]  + spheres[:100])


    # o3d.visualization.draw_geometries([annotation_pcd_ds])
    raise

