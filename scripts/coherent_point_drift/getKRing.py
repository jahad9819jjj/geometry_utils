import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from viz_curvature import load_ply_mesh


class CurvatureCalculator:
    def __init__(self, vertices, vertex_to_vertices):
        self.vertices = vertices
        self.vertex_to_vertices = vertex_to_vertices

    def getKRing(self, start, r):
        bufsize = self.vertices.shape[0]
        vv = []
        queue = [(start, 0)]
        visited = [False] * bufsize
        visited[start] = True

        while queue:
            toVisit, distance = queue.pop(0)
            vv.append((toVisit, distance))

            if toVisit < len(self.vertex_to_vertices):
                if distance < r:
                    for neighbor in self.vertex_to_vertices[toVisit]:
                        if not visited[neighbor]:
                            queue.append((neighbor, distance + 1))
                            visited[neighbor] = True

        return vv

# メッシュの読み込み
mesh = pv.read('/mnt/d/1_autoannotate/bcpd/data/Mug/remesh/Mug1_remesh.ply')
# mesh = pv.read('/mnt/d/1_autoannotate/bcpd/win/VisualStudio/BCPD-Win/debug/curv/X_mesh.ply')

verts, faces, curv = load_ply_mesh('/mnt/d/1_autoannotate/bcpd/win/VisualStudio/BCPD-Win/debug/curv/X_mesh.ply')
curv.PD3 = np.cross(curv.PD1, curv.PD2)
curv.PV3 = (curv.PV1 + curv.PV2) / 2

# TODO:各点に対する主曲率方向をLie群(SO(3))に修正し、mesh.point_data["lrf"]に格納
# TODO:各点に対する主曲率値（1,2,3）をmesh.point_data["pv{i}"]に格納

# 頂点とその隣接頂点の情報を取得
vertices = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]
vertex_to_vertices = [[] for _ in range(len(vertices))]
for face in faces:
    for i in range(3):
        vertex_to_vertices[face[i]].append(face[(i + 1) % 3])
        vertex_to_vertices[face[i]].append(face[(i + 2) % 3])

# CurvatureCalculator のインスタンスを作成
calculator = CurvatureCalculator(vertices, vertex_to_vertices)

# 始点と半径を指定して k-ring を取得
start_vertex = 0  # 始点となる頂点のインデックスを指定してください
radius = 50  # 半径を指定してください
k_ring = calculator.getKRing(start_vertex, radius)

# k レベルに応じて色を割り当て
colors = np.zeros((len(vertices), 3))
for vertex, level in k_ring:
    color = plt.cm.viridis(level / radius)[:3]
    # color = np.random.random(3)
    colors[vertex] = color
mesh.point_data['colors'] = colors

# メッシュを可視化
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='colors', rgb=True)
plotter.show()

# TODO:各点に対して、getKRingを計算し、各レベルごとのlrf(ある点に対する相対姿勢でも可)を取得（各レベルで等間隔にサンプリングしたlrfでもいい）
# TODO:各点に対して、lrfの分布をもつ
# TODO:計算した分布をもって、ターゲット点群に対してSinkhorn距離やKLなどの類似度計算をする