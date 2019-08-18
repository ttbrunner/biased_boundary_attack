"""
This code works with the "vanilla" ImageNet classification dataset, downloaded from http://image-net.org/download-images.
Tested with the ILSVRC2017 version.
"""

import numpy as np
import os

from scipy.io import loadmat
from scipy.misc import imresize
from scipy.ndimage import imread

from utils import imagenet_labels


def _load_labels(imagenet_base_path, limit):

    m_val_total = 50000
    m_val = m_val_total if limit is None else limit
    i_val_to_load = np.arange(m_val)

    # Somehow, TF Slim networks are trained with different clsids than the ones from the synset.
    # We therefore:
    # - Load the original synsets with descriptions
    # - Load the network clsids that we copypasted from somebody else who got it running (see imagenet_labels)
    # - Create a translation dict that maps from ILSVRC2015_CLSLOC_ID to the ids that we can use with the network.
    #
    # NOTE: One label ("crane") is duplicated, obviously with multiple meanings.
    #   Luckily, only one crane(label no. 134) is in this dataset. So we just hack it for now.
    #   Sorry for bad style.
    synsets = loadmat(os.path.join(imagenet_base_path, "devkit", "data", "meta_clsloc.mat"))['synsets'][0]
    assert len(synsets) == 1000 and len(synsets.shape) == 1
    clsloc_id_to_network_id = {}
    in_labels = {desc: i for i, desc in enumerate(imagenet_labels._lut)}
    in_labels["crane"] = 134
    for clsloc_id, data in enumerate(synsets):
        desc = data[2][0]
        network_id = in_labels[desc]
        clsloc_id_to_network_id[clsloc_id] = network_id

    # Get GT ids
    with open(os.path.join(imagenet_base_path, "devkit", "data", "ILSVRC2015_clsloc_validation_ground_truth.txt"), 'rt') as f:
        y_val = [int(l.strip()) - 1 for l in f.readlines()]         # shift 1-based ids to 0.
    y_val = [clsloc_id_to_network_id[i] for i in y_val]             # translate to the correct IDs.
    y_val = np.array(y_val)
    assert len(y_val) == m_val_total
    y_val = y_val[i_val_to_load]

    return y_val


def _load_img(path, shape):

    image_data = imread(path)
    image_data = imresize(image_data, shape, interp='bilinear')
    if len(image_data.shape) == 2:  # Grayscale
        image_data = np.repeat(image_data[:, :, np.newaxis], 3, axis=2)
    image_data = image_data[:, :, :3]

    return image_data


def load_dataset_y_val(imagenet_base_path, limit=None):
    """
    Loads all labels for the ImageNet validation set.
    """

    print('Loading dataset from source...')
    y_val = _load_labels(imagenet_base_path, limit=limit)

    return y_val


def load_on_demand_X_val(imagenet_base_path, indices):
    """
    Loads a range of images from the ImageNet validation set, specified by their indices.
    All images are in RGB uint8 format.
    """

    print("Lazy-loading {} imgs...".format(len(indices)))
    img_shape = (299, 299)
    imagenet_data_path = os.path.join(imagenet_base_path, "Data", "CLS-LOC")

    X_selected = []
    for i in indices:
        filepath = os.path.join(imagenet_data_path, "val", "ILSVRC2012_val_{:08d}.JPEG".format(i+1))
        X_selected.append(_load_img(filepath, img_shape))

    print("Done.")
    return np.array(X_selected)

