import gc
import os
import traceback

import numpy as np
import tensorflow as tf
from scipy.misc import imsave

import bench_settings
from attacks.biased_boundary_attack import BiasedBoundaryAttack
from utils.distance_measures import DistL2
from models.blackbox_wrapper import BlackboxWrapper
from models.imagenet_inception_v3.foolbox_model import create_imagenet_iv3_model
from models.imagenet_irn_v2.foolbox_model import create_imagenet_irn_v2_model
from utils import dataset_imagenet
from utils.imagenet_labels import label_to_name
from utils.sampling.sampling_provider import SamplingProvider
from utils.util import line_search_to_boundary, find_closest_img


def main():

    # Load dataset
    n_classes = 1000
    img_shape = (299, 299, 3)

    imagenet_base_path = "/path/to/ILSVRC2017/ILSVRC/"
    y_val = dataset_imagenet.load_dataset_y_val(imagenet_base_path, limit=None)

    # Run benchmark over entire ImageNet validation set. You may want to select a shuffled subset instead.
    y = y_val
    m = len(y)
    indices = np.arange(m)

    # Pick random target classes
    np.random.seed(0)
    y_target = np.random.randint(0, 1000, size=len(indices))

    with tf.Session() as sess:

        # Load model to attack and wrap it so we can be sure no info is leaking
        bb_model = BlackboxWrapper(create_imagenet_iv3_model(sess=sess))

        # Load surrogate model (if bias is enabled)
        surr_model = create_imagenet_irn_v2_model(sess=sess) if bench_settings.USE_SURROGATE_BIAS else None

        # Define distance measure. In its current form, the Boundary Attack uses L2.
        dm_l2 = DistL2().to_range_255()

        with SamplingProvider(shape=img_shape, n_threads=3, queue_lengths=80) as sample_gen:
            with BiasedBoundaryAttack(bb_model, sample_gen, dm_main=dm_l2, substitute_model=surr_model) as bba_attack:

                n_calls_max = 15000
                for i_img in range(m):
                    img_orig = dataset_imagenet.load_on_demand_X_val(imagenet_base_path, [indices[i_img]])[0]
                    clsid_gt = y[i_img]
                    clsid_target = y_target[i_img]

                    print("Image {}, original clsid={} ({}), target clsid={} ({}):".format(i_img, clsid_gt, label_to_name(clsid_gt),
                                                                                           clsid_target, label_to_name(clsid_target)))

                    img_log_dir_final = os.path.join("out_imagenet_bench{}".format(bench_settings.EXPERIMENT_SUFFIX), "{}".format(i_img))
                    img_log_dir = img_log_dir_final + ".inprog"

                    if os.path.exists(img_log_dir_final) or os.path.exists(img_log_dir):
                        continue
                    try:
                        os.makedirs(img_log_dir, exist_ok=False)

                        bb_model.adv_set_target(orig_img=img_orig, is_targeted=True, label=clsid_target, dist_measure=dm_l2,
                                                img_log_dir=img_log_dir, img_log_size=(299, 299), img_log_only_adv=True,
                                                print_calls_every=100)

                        # Starting point: Pick the closest image of the target class.
                        target_ids = np.arange(len(y_val))[y_val == clsid_target]
                        X_targets = dataset_imagenet.load_on_demand_X_val(imagenet_base_path, indices=target_ids)
                        X_start = find_closest_img(bb_model, X_orig=img_orig, X_targets=X_targets, label=clsid_target, is_targeted=True)

                        # First do a line search to boundary (save time)
                        X_start = line_search_to_boundary(bb_model, x_orig=img_orig, x_start=X_start, label=clsid_target, is_targeted=True)

                        # Now run the (biased) Boundary Attack.
                        bba_attack.run_attack(img_orig, label=clsid_target, is_targeted=True, X_start=X_start,
                                              n_calls_left_fn=(lambda: n_calls_max - bb_model.adv_get_n_calls()),
                                              source_step=2e-3, spherical_step=5e-2,
                                              mask=None, recalc_mask_every=(1 if bench_settings.USE_MASK_BIAS else None))

                        final_adversarial = bb_model.adv_get_best_img()

                        imsave(os.path.join(img_log_dir, "clean.png"), np.uint8(img_orig))
                        imsave(os.path.join(img_log_dir, "ae-final.png"), np.uint8(final_adversarial))
                        diff = np.float32(final_adversarial) - np.float32(img_orig) + 127.
                        imsave(os.path.join(img_log_dir, "diff.png"), np.uint8(np.round(diff)))

                        os.rename(img_log_dir, img_log_dir_final)

                        gc.collect()

                    except Exception:
                        # Log, but keep going.
                        print("Error trying to find adversarial example!")
                        traceback.print_exc()


if __name__ == "__main__":
    main()
