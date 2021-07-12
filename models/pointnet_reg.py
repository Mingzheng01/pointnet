import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name='pointcloud_input')
    rot_pl = tf.placeholder(tf.float32, shape=(batch_size, 3, 3), name='rotation_input')
    return pointclouds_pl, rot_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)

    return transform


def get_loss(pred, label, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    reg_loss = tf.nn.l2_loss(pred - label)
    #tf.summary.scalar('classify loss', reg_loss)

    # Enforce the transformation as orthogonal matrix
    K = pred.get_shape()[1].value
    mat_diff = tf.matmul(pred, tf.transpose(pred, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)

    mat_diff_sum = tf.reduce_sum(tf.abs(mat_diff), name='mat_diff_sum')

    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat_loss', mat_diff_loss)
    tf.summary.scalar('mat_reduce_sum', mat_diff_sum)

    return reg_loss + mat_diff_loss * 1., mat_diff_sum


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('logs', sess.graph)
            print(sess.run(outputs))
            writer.close()


