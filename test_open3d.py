import open3d as o3d
import numpy as np

x = np.linspace(-1, 1, 401)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.square(np.add(mesh_x, mesh_y))
#z_norm = (z - z.min()) / (z.max() - z.min())
xyz = np.zeros((np.size(mesh_x), 3))
xyz[:, 0] = np.reshape(mesh_x, -1)
xyz[:, 1] = np.reshape(mesh_y, -1)
xyz[:, 2] = np.reshape(z, -1)
print('xyz')
print(xyz)

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
colors = xyz
colors[:, 0] = colors[:, 2]
colors[:, 1] = colors[:, 2]
colors = colors / (np.max(colors[:, 0]) - np.min(colors[:, 0]))
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd, axes])

print(xyz[:, 2])
