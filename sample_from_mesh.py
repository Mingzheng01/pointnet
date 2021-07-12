import open3d as o3d
import os
import numpy as np
import h5py

point_clouds = []
for dirpath, dirnames, filenames in os.walk("F:\\cases\\tooth_11_stls"):
    for filename in filenames:
        print(os.path.splitext(filename)[-1])
        if os.path.splitext(filename)[-1] != ".stl":
            continue

        full_filename = os.path.join(dirpath, filename)
        mesh = o3d.io.read_triangle_mesh(full_filename)
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        print(mesh)
        pcd = mesh.sample_points_poisson_disk(1024)
        print(pcd)
        #o3d.visualization.draw_geometries([mesh, pcd], mesh_show_wireframe=True)
        #base_name = os.path.splitext(os.path.basename(filename))[0]
        #o3d.io.write_point_cloud(os.path.join(dirpath, base_name) + ".ply", pcd)
        point_clouds.append(np.array(pcd.points))

f = h5py.File("F:\\cases\\tooth_11_stls\\point_clouds.hdf5", mode='w')
f["point_clouds"] = point_clouds
f.close()

