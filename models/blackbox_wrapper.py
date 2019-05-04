import os
import numpy as np
from scipy.misc import imsave, imresize
from foolbox.models import Model
import yaml

from utils.distance_measures import DistanceMeasure


class BlackboxWrapper(Model):
    """
    This wrapper turns a white-box Foolbox model into a label-only black box.
    - Rounds&clips every image before sending it to the model, and converts logits to one-hot before returning the result
    - Keeps track of the best adversarial example, and number of queries
    - Can log all queried images to an output dir.
    """

    def __init__(self, model):
        super(BlackboxWrapper, self).__init__(
            bounds=model.bounds(),
            channel_axis=model.channel_axis())

        self.wrapped_model = model

        self._adv_orig_img = None
        self._adv_is_targeted = False
        self._adv_label = None
        self._adv_dist_measure = None
        self._adv_best_img = None
        self._adv_best_dist = None
        self._adv_n_calls = 0               # Count number of calls since the last adversarial target was set.
        self._adv_print_calls_every = None
        self._adv_candidate_stats = None

        # For log output of query images.
        self._img_log_dir = None
        self._img_log_only_adv = False
        self._img_log_size = (299, 299)

    def __enter__(self):
        assert self.wrapped_model.__enter__() == self.wrapped_model
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped_model.__exit__(exc_type, exc_value, traceback)

    def num_classes(self):
        return self.wrapped_model.num_classes()

    def batch_predictions(self, images):
        """
        Runs a batch prediction on the model. For convenience, use predictions() instead.
        :param images: (batch, height, width, channels)
        :return: One-hot prediction vector (batch, n_classes), rounded to 1/0.
        """

        # Round & clip the image to uint8.
        img_batch_to_eval = np.uint8(np.round(np.clip(images, 0, 255)))

        preds = self.wrapped_model.batch_predictions(img_batch_to_eval)
        clsids = np.argmax(preds, axis=1)

        # Clean up the preds (1 0 0 0 etc), to make sure the attack can't use logits.
        preds_cleaned = np.zeros_like(preds)
        preds_cleaned[np.arange(len(preds)), clsids] = 1.

        # Go through all imgs in the batch and check if adversarial
        if self._adv_orig_img is not None:
            for i in range(img_batch_to_eval.shape[0]):
                self._adv_n_calls += 1
                if self._adv_print_calls_every is not None and self._adv_n_calls % self._adv_print_calls_every == 0:
                    print("{} queries elapsed.".format(self._adv_n_calls))

                is_adv = bool(self._adv_is_targeted == (clsids[i] == self._adv_label))
                if is_adv:
                    dist = self._adv_dist_measure.calc(self._adv_orig_img, img_batch_to_eval[i])
                    if dist < self._adv_best_dist:
                        self._adv_best_dist = dist
                        self._adv_best_img = img_batch_to_eval[i]

                if self._img_log_dir is not None:
                    self.log_img(img_batch_to_eval[i], is_adv=is_adv)

        return preds_cleaned

    def adv_set_target(self, orig_img, is_targeted, label, dist_measure: DistanceMeasure,
                       img_log_dir=None, img_log_only_adv=False, img_log_size=(299, 299), n_prev_calls=0, print_calls_every=None):
        """
        Sets a new adversarial target and resets the n_calls counter.
        Call this once for every new image.
        :param orig_img: The original image being perturbed
        :param is_targeted: If true, then the attack is successful if pred==label.
                            If false, then the attack is succesful if pred!=label.
        :param label: If is_targeted is True, then the target adversarial class.
                      If is_targeted is False, then the original (correct) class.
        :param dist_measure: Distance measure to use.
        :param img_log_dir: Optional: Directory to which requested images are logged.
                            For each query, the image and a YAML file with metadata of the attack is logged.
        :param img_log_only_adv: If true, only successful(adversarial) queries are logged. If false, all queries are logged.
        :param img_log_size: Resolution of logged images.
        :param n_prev_calls: Optional: Initialize the query counter with this number (e.g. if continuing a previously saved attack).
        :param print_calls_every: Optional: Prints the number of elapsed queries every (n) queries.

        """
        self._adv_orig_img = orig_img
        self._adv_is_targeted = is_targeted
        self._adv_label = label
        self._adv_dist_measure = dist_measure.to_range_255()
        self._adv_best_img = None
        self._adv_best_dist = 99999.
        self._adv_n_calls = n_prev_calls
        self._adv_print_calls_every = print_calls_every
        self._adv_candidate_stats = None

        self._img_log_dir = img_log_dir
        self._img_log_only_adv = img_log_only_adv
        self._img_log_size = img_log_size

    def adv_get_best_img(self):
        return self._adv_best_img

    def adv_get_n_calls(self):
        return self._adv_n_calls

    def adv_set_stats(self, candidate_stats):
        # Registers stats for the next image. Use this before predict/batch_predict. Once the model query is made, log_img will log these
        #  stats into a yaml file.
        self._adv_candidate_stats = candidate_stats

    def log_img(self, img, is_adv):
        assert img.dtype == np.uint8
        assert self._adv_orig_img is not None, "Need to initialize adversarial target when logging images!"

        if self._img_log_only_adv and not is_adv:
            return

        filepath = os.path.join(self._img_log_dir, "img-{:05d}.png".format(self._adv_n_calls))
        img = imresize(img, self._img_log_size, interp='nearest')
        imsave(filepath, img)

        # diff = np.float32(img) - np.float32(self._adv_orig_img)
        # diff = np.uint8(np.round(np.clip(127. + diff, 0, 255)))
        # diff_filepath = os.path.join(self._img_log_dir, "diff-latest.png".format(self._adv_n_calls))
        # diff = imresize(diff, self._img_log_size, interp='nearest')
        # imsave(diff_filepath, diff)

        # All meta
        metadata = {}
        if self._adv_candidate_stats is not None:
            metadata.update(self._adv_candidate_stats)
            self._adv_candidate_stats = None

        # Note: need to explicitly cast numpy scalars to float or int etc, otherwise the object gets serialized.
        metadata["dist"] = float(self._adv_dist_measure.calc(img, self._adv_orig_img))
        metadata["is_adv"] = is_adv
        meta_filepath = os.path.join(self._img_log_dir, "meta-{:05d}.yaml".format(self._adv_n_calls))
        with open(meta_filepath, 'w') as outfile:
            yaml.dump(metadata, outfile)

