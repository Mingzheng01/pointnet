import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import pointnet_reg as MODEL
import h5py
import numpy as np
import pc_util as util
import open3d as o3d
import scipy
from scipy.spatial.transform import Rotation

# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
sess = tf.Session(config=config)

new_saver = tf.train.import_meta_graph('log/reg_model-49.meta')

# Restore variables from disk.
new_saver.restore(sess, 'log/reg_model-49')

file = h5py.File("./data/tooth_11_point_clouds.h5", 'r')
print(file.keys())

point_cloud_dst = file['pointclouds']
print(point_cloud_dst)
point_clouds = []
rot_matrix = []
for i in range(point_cloud_dst.shape[0]):
    point_clouds.append(np.array(point_cloud_dst[i, 0 : 1024, :]))
point_clouds = np.array(point_clouds)


prediction = tf.compat.v1.get_default_graph().get_tensor_by_name("transform_net1/prediction:0")
input = tf.compat.v1.get_default_graph().get_tensor_by_name("pointcloud_input:0")
is_training = tf.compat.v1.get_default_graph().get_tensor_by_name("Placeholder:0")

first_batch_pc = np.copy(point_clouds[0:32, :, :])

ret = sess.run(prediction, feed_dict={input:first_batch_pc, is_training:False})
ret_trans = np.transpose(ret, [0, 2, 1])

print(ret_trans)

first_batch_pc_res = np.copy(point_clouds[0:32, :, :])
for i in range(32):
    for j in range(1024):
        p = first_batch_pc_res[i, j ,:]
        r = np.matmul(ret_trans[i, :, :], p)
        first_batch_pc_res[i, j, :] = r

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(first_batch_pc_res[i, :, :])
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.)
    o3d.visualization.draw_geometries([pcd, axes])










