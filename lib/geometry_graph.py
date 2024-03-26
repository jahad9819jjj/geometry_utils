import numpy as np
from collections import defaultdict
import heapq

class SpatialGraph:
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles
        self.graph = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        for triangle in self.triangles:
            for i in range(3):
                v1 = triangle[i]
                v2 = triangle[(i + 1) % 3]
                dist = self._distance(self.vertices[v1], self.vertices[v2])
                self.graph[v1].append((v2, dist))
                self.graph[v2].append((v1, dist))

    def _distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def get_neighbors(self, v):
        return self.graph[v]

    def k_nearest_neighbors(self, query, k):
        distances = []
        for i, vertex in enumerate(self.vertices):
            dist = self._distance(query, vertex)
            heapq.heappush(distances, (dist, i))

        k_nearest_vertices = []
        k_nearest_distances = []
        for _ in range(k):
            if distances:
                dist, idx = heapq.heappop(distances)
                k_nearest_vertices.append(self.vertices[idx])
                k_nearest_distances.append(dist)
            else:
                break

        k_nearest_vertices = np.array(k_nearest_vertices)
        k_nearest_distances = np.array(k_nearest_distances)

        return k_nearest_vertices, k_nearest_distances
    
    def self_nearest_neighbors(self, query, k):
        # TODO: implement effective algorithm like dijkstra method
        # クエリ点が既存の頂点と一致するかどうかを確認
        vertex_indices = np.where((self.vertices == query).all(axis=1))[0]

        if len(vertex_indices) > 0:
            # クエリ点が既存の頂点と一致する場合
            nearest_vertices = self.vertices[vertex_indices[:k]]
            raise NotImplementedError(f'Not implemented yet.')
        else:
            # クエリ点が既存の頂点と一致しない場合
            distances = []
            for i, vertex in enumerate(self.vertices):
                dist = self._distance(query, vertex)
                heapq.heappush(distances, (dist, i))

            nearest_vertices = []
            nearest_distances = []
            for _ in range(k):
                if distances:
                    dist, idx = heapq.heappop(distances)
                    nearest_vertices.append(self.vertices[idx])
                    nearest_distances.append(dist)
                else:
                    break

            nearest_vertices = np.array(nearest_vertices)
            nearest_distances = np.array(nearest_distances)

        return nearest_vertices, nearest_distances
    

if __name__ == "__main__":
    import pyvista
    from pyvista import examples
    bunny = examples.download_bunny()
    vertices = bunny.points
    triangles = bunny.faces.reshape(-1, 4)[:, 1:]

    q = vertices[0]
    sg = SpatialGraph(vertices, triangles)
    nearest_neighbors, nearest_distances  = sg.k_nearest_neighbors(q, k=100)


    plotter = pyvista.Plotter()
    plotter.add_mesh(bunny, color="white", show_edges=True, opacity=0.5)
    plotter.add_points(q, color="red", point_size=20)
    plotter.add_points(nearest_neighbors, color="blue", point_size=20)
    for i in range(len(nearest_neighbors)):
        line = pyvista.Line(q, nearest_neighbors[i])
        plotter.add_mesh(line, color="green", line_width=2)
    plotter.show()