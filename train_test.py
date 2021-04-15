import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import pointnet_cls as MODEL

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
MAX_EPOCH = 50

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

def train(point_cloud_data, point_cloud_labels):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(0)):
            #get the place holders
            point_clouds_ph, labels_ph = MODEL.placeholder_inputs(32, 1024)

            # is training place holder..
            is_training_ph = tf.placeholder(tf.bool, shape=())
            print(is_training_ph)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn decay', bn_decay)
            print(bn_decay)

            #get model and loss
            pred, end_points = MODEL.get_model(point_clouds_ph, is_training_ph, bn_decay)
            loss = MODEL.get_loss(pred, labels_ph, end_points)
            tf.summary.scalar("loss", loss)
            print(pred)
            print(loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_ph))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32) / float(BATCH_SIZE))
            tf.summary.scalar("accuracy", accuracy)

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
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "test_test"))

        # init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_ph:True})

        ops = {'pointclouds_pl': point_clouds_ph,
               'labels_pl': labels_ph,
               'is_training_pl': is_training_ph,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'point_cloud_data' : point_cloud_data,
               'point_cloud_labels':point_cloud_labels}

        for epoch in range(MAX_EPOCH):
            log_string('-------------- EPOCH %03d ---------------------' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)

def train_one_epoch(sess, ops, train_writer):

    is_training = True

    print(ops['point_cloud_data'].shape)
    print(ops['point_cloud_labels'].shape)

    point_clouds = ops['point_cloud_data']
    labels = ops['point_cloud_labels']

    data_size = ops['point_cloud_data'].shape[0]
    nums = np.array(range(data_size))

    num_batches = data_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        batch_indices = np.random.choice(nums, BATCH_SIZE)

        selected_point_clouds = point_clouds[batch_indices]
        selected_labels = labels[batch_indices]

        rotated_data = provider.rotate_point_cloud(selected_point_clouds)
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: selected_labels,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                         ops['train_op'], ops['loss'], ops['pred']],
                                                        feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == selected_labels)
        total_correct += correct
        total_seen += BATCH_SIZE

        loss_sum += loss_val

    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == "__main__":

    file = h5py.File("./data/upper_and_lower_5_cls.h5", 'r')
    print(file.keys())

    point_cloud_dst = file['pointclouds']
    print(point_cloud_dst)


    point_clouds = []
    point_cloud_labels = []
    for i in range(point_cloud_dst.shape[0]):
        point_clouds.append(np.array(point_cloud_dst[i, 1 : 1025, :]))
        point_cloud_labels.append(point_cloud_dst[i, 0, 0])

    point_clouds = np.array(point_clouds)
    point_cloud_labels = np.array(point_cloud_labels).astype(int)
    print(point_clouds.shape)
    print(point_cloud_labels.shape)

    train(point_clouds, point_cloud_labels)
