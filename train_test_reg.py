import h5py
import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import pointnet_reg as MODEL
import vg
import open3d as o3d
import scipy
from scipy.spatial.transform import Rotation

DECAY_STEP = 2
DECAY_RATE = 0.7
BASE_LEARNING_RATE = 0.001
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
MOMENTUM = 0.9
BATCH_SIZE = 32
LOG_DIR = 'log'
MAX_EPOCH = 80

def log_string(out_str):
   # LOG_FOUT.write(out_str+'\n')
    #LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(point_cloud_data):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(0)):
            #get the place holders
            point_clouds_ph, rot_ph = MODEL.placeholder_inputs(32, 1024)

            # is training place holder..
            is_training_ph = tf.placeholder(tf.bool, shape=())
            print(is_training_ph)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn decay', bn_decay)
            print(bn_decay)

            #get model and loss
            pred = MODEL.get_model(point_clouds_ph, is_training_ph, bn_decay)
            loss, mat_diff_sum = MODEL.get_loss(pred, rot_ph)
            tf.summary.scalar("loss", loss)
            print(pred)
            print(loss)

            #correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(rot_ph))
            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32) / float(BATCH_SIZE))
            #tf.summary.scalar("accuracy", accuracy)

            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning rate', learning_rate)

            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            train_op = optimizer.minimize(loss, global_step=batch)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config = config)

        # add summary writers..
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "train_test"), sess.graph)

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_ph:True})

        ops = {'pointclouds_pl': point_clouds_ph,
               'rot_ph': rot_ph,
               'is_training_pl': is_training_ph,
               'pred': pred,
               'loss': loss,
               "mat_diff_sum":mat_diff_sum,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'point_cloud_data' : point_cloud_data}

        for epoch in range(MAX_EPOCH):
            log_string('-------------- EPOCH %03d ---------------------' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)


            save_path = saver.save(sess, os.path.join(LOG_DIR, "reg_model"), global_step=epoch )
            log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):

    is_training = True
    point_clouds = ops['point_cloud_data']

    data_size = ops['point_cloud_data'].shape[0]
    nums = np.array(range(data_size))

    num_batches = data_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    rand_rot_mat = []
    for _ in range(point_clouds.shape[0]):
        # mat = np.random.rand(3, 3)
        #q, _ = np.linalg.qr(mat)
        #rand_rot_mat.append(q)
        mat = scipy.spatial.transform.Rotation.random().as_matrix()
        rand_rot_mat.append(np.array(mat))
    rand_rot_mat = np.array(rand_rot_mat)

    for batch_idx in range(num_batches):
        batch_indices = np.random.choice(nums, BATCH_SIZE)

        selected_point_clouds = point_clouds[batch_indices]
        selected_rot_mat = rand_rot_mat[batch_indices]

        rotated_point_clouds = selected_point_clouds
        for i in range(BATCH_SIZE):
            for j in range(selected_point_clouds.shape[1]):
                rotated_point_clouds[i, j, :] = np.matmul(selected_rot_mat[i], selected_point_clouds[i, j, :])

        #jittered_data = provider.jitter_point_cloud(rotated_data)

        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(point_clouds[0])
        #axes= o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.)
       # axes.rotate(np.array(selected_rot_mat[0]))
        #o3d.visualization.draw_geometries([pcd, axes])

        feed_dict = {ops['pointclouds_pl']: rotated_point_clouds,
                     ops['rot_ph']: selected_rot_mat,
                     ops['is_training_pl']: is_training, }

        writer = tf.summary.FileWriter('logs', sess.graph)
        summary, step, _, loss_val, pred_val, mat_diff_sum = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred'],ops['mat_diff_sum']],
                                                        feed_dict=feed_dict)
       # writer.add_summary(summary=mat_reduce_sum, global_step=0)
        writer.close()

        rot_vector_pred = np.array([np.matmul(pred_val[i, :, :], [1., 1., 1.]) for i in range(BATCH_SIZE)])
        rot_vector = np.array([np.matmul(selected_rot_mat[i, :, :], [1., 1., 1.]) for i in range(BATCH_SIZE)])
        #for i in range(BATCH_SIZE):
            #print(vg.angle(rot_vector[i], rot_vector_pred[i]))

        train_writer.add_summary(summary, step)
        #pred_val = np.argmax(pred_val, 1)
        #correct = np.sum(pred_val == selected_labels)
        #total_correct += correct
        total_seen += BATCH_SIZE

        loss_sum += loss_val

        print("batch loss: %f" % loss_val)
        print("mat diff sum: %f" % mat_diff_sum)

        prediction = tf.compat.v1.get_default_graph().get_tensor_by_name("transform_net1/prediction:0")
        input = tf.compat.v1.get_default_graph().get_tensor_by_name("pointcloud_input:0")
        m_is_training = tf.compat.v1.get_default_graph().get_tensor_by_name("Placeholder:0")

        m_ret = sess.run(prediction, feed_dict={input:rotated_point_clouds, m_is_training:False})
        print(m_ret[0])
        print(selected_rot_mat[0])

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == "__main__":

    file = h5py.File("F:\\cases\\tooth_11_stls\\point_clouds.hdf5", 'r')
    print(file.keys())

    point_clouds = file['point_clouds']
    print(point_clouds)

    point_clouds = np.array(point_clouds)


    print(point_clouds.shape)

    train(point_clouds)
