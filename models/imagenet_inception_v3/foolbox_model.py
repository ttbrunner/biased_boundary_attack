"""
This is InceptionV3 trained in ImageNet (see imagenet_inception_v3.py)
"""
import tensorflow as tf
from foolbox.models import TensorFlowModel

from models.imagenet_inception_v3.imagenet_inception_v3 import model


def create_imagenet_iv3_model(sess=None, x_input=None):

    # Allow to reuse a session
    if sess is None:
        sess = tf.Session()

    # Allow to put model on top of an existing input
    if x_input is None:
        x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))

    # NOTE: don't norm here. For some reason this InceptionV3 accepts preproc only in specific scopes,
    # so it is done in imagenet_inception_v3.py.

    logits, preds = model(sess=sess, image=x_input)

    with sess.as_default():
        fmodel = TensorFlowModel(x_input, logits, bounds=(0, 255))
    return fmodel
