# Inception v3 trained on ImageNet.
#
# Adapted from https://github.com/labsix/limited-blackbox-attacks
# Ilyas et al. "Black-Box Adversarial Attacks with Limited Queries and Information"


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools
import os

SIZE = 299

_INCEPTION_CHECKPOINT_NAME = 'inception_v3.ckpt'
INCEPTION_CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    'checkpoints',
    _INCEPTION_CHECKPOINT_NAME
)


def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3

    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn


def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        image /= 255.
        # image = tf.image.resize_bilinear(image, [height, width], align_corners=False)   # No need, inputs are always in target shape.
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


# input is [batch, 299, 299, 3], pixels in [0, 1]
# output is [batch, 1000]
_inception_initialized = False
def model(sess, image):
    global _inception_initialized
    network_fn = _get_model(reuse=_inception_initialized)
    size = network_fn.default_image_size
    preprocessed = _preprocess(image, size, size)
    logits, _ = network_fn(preprocessed)
    logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _inception_initialized:
        optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        # saver = tf.train.Saver()
        # saver.restore(sess, INCEPTION_CHECKPOINT_PATH)
        _inception_initialized = True

    return logits, predictions