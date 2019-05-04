"""
This is InceptionResnetV2 trained on ImageNet (see imagenet_irn_v2.py)
"""
import os
import tensorflow as tf
from foolbox.models import TensorFlowModel

from models.imagenet_irn_v2 import inception_resnet_v2


def create_imagenet_irn_v2_model(sess=None, x_input=None, use_adv_trained_tramer=False):

    # Allow to reuse a session and put the model on top of an existing input
    if sess is None:
        sess = tf.Session()

    vars_before = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    with sess.graph.as_default():
        if x_input is None:
            x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))

        with tf.contrib.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                x_input, num_classes=1001, is_training=False, create_aux_logits=True)
            logits = logits[:, 1:]  # ignore background class

    vars_after = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
    vars_new = set(vars_after).difference(vars_before)

    with tf.variable_scope('utilities'):
        saver = tf.train.Saver(list(vars_new))

    path = os.path.dirname(os.path.abspath(__file__))
    if use_adv_trained_tramer:
        path = os.path.join(path, 'checkpoints', 'ens_adv_inception_resnet_v2.ckpt')
    else:
        path = os.path.join(path, 'checkpoints', 'inception_resnet_v2_2016_08_30.ckpt')

    with sess.graph.as_default():
        saver.restore(sess, path)  # tf.train.latest_checkpoint(path))

    with sess.as_default():
        fmodel = TensorFlowModel(x_input, logits, bounds=(0, 255))
    return fmodel
