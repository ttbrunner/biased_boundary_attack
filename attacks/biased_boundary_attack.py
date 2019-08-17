import numpy as np
import timeit
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor

from foolbox.models import TensorFlowModel

import bench_settings
from models.batch_tensorflow_model import BatchTensorflowModel
from utils.util import sample_hypersphere


class BiasedBoundaryAttack:
    """
     Like BoundaryAttack, but uses biased sampling from various sources.
     This implementation is optimized for speed: it can query the model and, while waiting, already prepare the next perturbation candidate.
     Ideally, there is zero overhead over the prediction speed of the model under attack.

     However, we do NOT run predictions in parallel (as the Foolbox BoundaryAttack does).
     This attack is completely sequential to keep the number of queries minimal.

     Activate various biases in bench_settings.py.
    """

    def __init__(self, blackbox_model, sample_gen, dm_main, substitute_model=None):
        """
        Creates an instance that can be reused when attacking multiple images.
        :param blackbox_model: The model to attack.
        :param sample_gen: Random sample generator.
        :param substitute_model: A Foolbox differentiable surrogate model for gradients. If None, then the surrogate bias will not be used.
        """

        self.blackbox_model = blackbox_model
        self.sample_gen = sample_gen
        self.dm_main = dm_main.to_range_01()          # Images are normed to 0/1 inside of run_attack()

        # A substitute model that provides batched gradients.
        if substitute_model is not None:
            if not isinstance(substitute_model, TensorFlowModel):
                raise ValueError("Substitute Model must provide gradients! (Foolbox: TensorFlowModel)")
            self.substitute_model = BatchTensorflowModel(substitute_model._images, substitute_model._batch_logits, session=substitute_model.session)
        else:
            self.substitute_model = None

        # We use ThreadPools to calculate candidates and surrogate gradients while we're waiting for the model's next prediction.
        self.pg_thread_pool = ThreadPoolExecutor(max_workers=1)
        self.candidate_thread_pool = ThreadPoolExecutor(max_workers=1)

    def __enter__(self):
        self.pg_thread_pool.__enter__()
        self.candidate_thread_pool.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Will block until the futures are calculated. Thankfully they're not very complicated.
        self.pg_thread_pool.__exit__(exc_type, exc_value, traceback)
        self.candidate_thread_pool.__exit__(exc_type, exc_value, traceback)
        print("BiasedBoundaryAttack: all threads stopped.")

    def run_attack(self, X_orig, label, is_targeted, X_start, n_calls_left_fn, n_max_per_batch=50, n_seconds=None,
                   source_step=1e-2, spherical_step=1e-2, mask=None, recalc_mask_every=None):
        """
        Runs the Biased Boundary Attack against a single image.
        The attack terminates when n_calls_left_fn() returns 0 or n_seconds have elapsed.

        :param X_orig: The original (clean) image to perturb.
        :param label: The target label (if targeted), or the original label (if untargeted).
        :param is_targeted: True if targeted.
        :param X_start: The starting point (must be of target class).
        :param n_calls_left_fn: A function that returns the currently remaining number of queries against the model.
        :param n_max_per_batch: How many samples are drawn per "batch". Samples are processed serially (the challenge doesn't allow
                                batching), but for each sample, the attack dynamically adjusts hyperparams based on the success of
                                previous samples. This "batch" size is the max number of samples after which hyperparams are reset, and
                                a new "batch" is started. See generate_candidate().
        :param n_seconds: Maximum seconds allowed for the attack to complete.
        :param source_step: source step hyperparameter (see Boundary Attack)
        :param spherical_step: orthogonal step hyperparameter (see Boundary Attack)
        :param mask: Optional. If not none, a predefined mask (expert knowledge?) can be defined that will be applied to the perturbations.
        :param recalc_mask_every: If not none, automatically calculates a mask from the current image difference.
                                  Will recalculate this mask every (n) steps.
        :return: The best adversarial example so far.
        """

        assert len(X_orig.shape) == 3
        assert len(X_start.shape) == 3
        if mask is not None:
            assert mask.shape == X_orig.shape
            assert np.sum(mask < 0) == 0 and 1. - np.max(mask) < 1e-4, "Mask must be scaled to [0,1]. At least one value must be 1."
        else:
            mask = np.ones(X_orig.shape, dtype=np.float32)

        time_start = timeit.default_timer()

        pg_future = None
        try:
            # WARN: Inside this function, image space is normed to [0,1]!
            X_orig = np.float32(X_orig) / 255.
            X_start = np.float32(X_start) / 255.

            label_current, dist_best = self._eval_sample(X_start, X_orig)
            if (label_current == label) != is_targeted:
                print("WARN: Starting point is not a valid adversarial example! Continuing for now.")

            X_adv_best = np.copy(X_start)

            last_mask_recalc_calls = n_calls_left_fn()

            # Abort if we're running out of queries
            while n_calls_left_fn() > 3:

                # Mask Bias: recalculate mask from current diff (hopefully this reduces the search space)
                if recalc_mask_every is not None and last_mask_recalc_calls - n_calls_left_fn() >= recalc_mask_every:
                    new_mask = np.abs(X_adv_best - X_orig)
                    new_mask /= np.max(new_mask)                     # scale to [0,1]
                    new_mask = new_mask ** 0.5                       # weaken the effect a bit.
                    print("Recalculated mask. Weighted dimensionality of search space is now {:.0f} (diff: {:.2%}). ".format(
                           np.sum(new_mask), 1. - np.sum(new_mask) / np.sum(mask)))
                    mask = new_mask
                    last_mask_recalc_calls = n_calls_left_fn()

                # Draw n candidates at the current position (before resetting hyperparams or before reaching the limit)
                n_candidates = min(n_max_per_batch, n_calls_left_fn())

                # Calculate the projected adversarial surrogate gradient at the current position.
                #  Putting this into a ThreadPoolExecutor. While this is processing, we can already start drawing the first sample.
                # Also cancel any pending requests from previous steps.
                if pg_future is not None:
                    pg_future.cancel()
                pg_future = self.pg_thread_pool.submit(self.get_projected_gradients, **{
                    "x_current": X_adv_best,
                    "x_orig": X_orig,
                    "label": label,
                    "is_targeted": is_targeted})

                # Also do candidate generation with a ThreadPoolExecutor.
                # Queue the first candidate.
                candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                    "i": 0,
                    "n": n_candidates,
                    "x_orig": X_orig,
                    "x_current": X_adv_best,
                    "mask": mask,
                    "source_step": source_step,
                    "spherical_step": spherical_step,
                    "pg_future": pg_future})

                for i in range(n_candidates):
                    # Get candidate and queue the next one.
                    candidate, stats = candidate_future.result()
                    if i < n_candidates - 1:
                        candidate_future = self.candidate_thread_pool.submit(self.generate_candidate, **{
                            "i": i+1,
                            "n": n_candidates,
                            "x_orig": X_orig,
                            "x_current": X_adv_best,
                            "mask": mask,
                            "source_step": source_step,
                            "spherical_step": spherical_step,
                            "pg_future": pg_future})

                    time_elapsed = timeit.default_timer() - time_start
                    if n_seconds is not None and time_elapsed >= n_seconds:
                        print("WARN: Running out of time! Aborting attack!")
                        return X_adv_best * 255.

                    # Test if successful. NOTE: dist is rounded here!
                    self.blackbox_model.adv_set_stats(stats)
                    candidate_label, rounded_dist = self._eval_sample(candidate, X_orig)
                    unrounded_dist = self.dm_main.calc(candidate, X_orig)
                    if (candidate_label == label) == is_targeted:
                        if unrounded_dist < dist_best:
                            print("@ {:.3f}: After {} samples, found something @ {:.3f} (rounded {:.3f})! (reduced by {:.1%})".format(
                                dist_best, i, unrounded_dist, rounded_dist, 1.-unrounded_dist/dist_best))

                            # Terminate this batch (don't try the other candidates) and advance.
                            X_adv_best = candidate
                            dist_best = unrounded_dist
                            break

            return X_adv_best * 255.

        finally:
            # Be safe and wait for the gradient future. We want to be sure that no BG worker is blocking the GPU before returning.
            if pg_future is not None:
                futures.wait([pg_future])

    def generate_candidate(self, i, n, x_orig, x_current, mask, source_step, spherical_step, pg_future):

        # This runs in a loop (while i<n) per "batch".
        # Whenever a candidate is successful, a new batch is started. Therefore, i is the number of previously unsuccessful samples.
        # Trying to use this in our favor, we progressively reduce step size for the next candidate.
        # When the batch is through, hyperparameters are reset for the next batch.

        # Scale both spherical and source step with i.
        scale = (1. - i/n) + 0.3
        c_source_step = source_step * scale
        c_spherical_step = spherical_step * scale

        # Get the adversarial projected gradient from the (other) BG worker.
        pg_factor = 0.3
        pgs = pg_future.result()
        pgs = pgs if i % 2 == 0 else None           # Only use gradient bias on every 2nd iteration, but always try it at first..

        if bench_settings.USE_PERLIN_BIAS:
            sampling_fn = self.sample_gen.get_perlin
        else:
            sampling_fn = self.sample_gen.get_normal

        candidate, stats = self.generate_boundary_sample(
            X_orig=x_orig, X_adv_current=x_current, mask=mask, source_step=c_source_step, spherical_step=c_spherical_step,
            sampling_fn=sampling_fn, pgs_current=pgs, pg_factor=pg_factor)

        stats["i_sample"] = int(i)
        stats["mask_sum"] = float(np.sum(mask))
        return candidate, stats

    def generate_boundary_sample(self, X_orig, X_adv_current, mask, source_step, spherical_step, sampling_fn,
                                 pgs_current=None, pg_factor=0.5):
        # Adapted from FoolBox BoundaryAttack.

        unnormalized_source_direction = np.float32(X_orig) - np.float32(X_adv_current)
        source_norm = np.linalg.norm(unnormalized_source_direction)
        source_direction = unnormalized_source_direction / source_norm

        # Get perturbation from provided distribution
        sampling_dir, stats = sampling_fn(return_stats=True)

        # ===========================================================
        # calculate candidate on sphere
        # ===========================================================
        dot = np.vdot(sampling_dir, source_direction)
        sampling_dir -= dot * source_direction                                      # Project orthogonal to source direction
        sampling_dir *= mask                                                        # Apply regional mask
        sampling_dir /= np.linalg.norm(sampling_dir)                                # Norming increases magnitude of masked regions

        # If available: Bias the spherical dirs in direction of the adversarial gradient, which is projected onto the sphere
        if pgs_current is not None:
            # We have a bunch of gradients that we can try. Randomly select one.
            # NOTE: we found this to perform better than simply averaging the gradients.
            pg_current = pgs_current[np.random.randint(0, len(pgs_current))]
            pg_current *= mask
            pg_current /= np.linalg.norm(pg_current)

            sampling_dir = (1. - pg_factor) * sampling_dir + pg_factor * pg_current
            sampling_dir /= np.linalg.norm(sampling_dir)

        sampling_dir *= spherical_step * source_norm                                # Norm to length stepsize*(dist from src)

        D = 1 / np.sqrt(spherical_step ** 2 + 1)
        direction = sampling_dir - unnormalized_source_direction
        spherical_candidate = X_orig + D * direction

        np.clip(spherical_candidate, 0., 1., out=spherical_candidate)

        # ===========================================================
        # step towards source
        # ===========================================================
        new_source_direction = X_orig - spherical_candidate

        new_source_direction_norm = np.linalg.norm(new_source_direction)
        new_source_direction /= new_source_direction_norm
        spherical_candidate = X_orig - source_norm * new_source_direction           # Snap sph.c. onto sphere

        # From there, take a step towards the target.
        candidate = spherical_candidate + (source_step * source_norm) * new_source_direction

        np.clip(candidate, 0., 1., out=candidate)
        return np.float32(candidate), stats

    def get_projected_gradients(self, x_current, x_orig, label, is_targeted):
        # Idea is: we have a direction (spherical candidate) in which we want to sample.
        # We know that the gradient of a substitute model, projected onto the sphere, usually points to an adversarial region.
        # Even if we are already adversarial, it should point "deeper" into that region.
        # If we sample in that direction, we should move toward the center of the adversarial cone.
        # Here, we simply project the gradient onto the same hyperplane as the spherical samples.
        #
        # Instead of a single projected gradient, this method returns an entire batch of them:
        # - Surrogate gradients are unreliable, so we sample them in a region around the current position.
        # - This gives us a similar benefit as observed in "PGD with random restarts".

        if self.substitute_model is None:
            return None

        source_direction = x_orig - x_current
        source_norm = np.linalg.norm(source_direction)
        source_direction = source_direction / source_norm

        # Take a tiny step towards the source before calculating the gradient. This marginally improves our results.
        step_inside = 1e-2 * source_norm
        x_inside = x_current + step_inside * source_direction

        # Perturb the current position before calc'ing gradient
        n_samples = 4
        radius_max = 1e-2 * source_norm   
        x_perturb = sample_hypersphere(n_samples=n_samples, sample_shape=x_orig.shape, radius=1, sample_gen=self.sample_gen)
        x_perturb *= np.random.uniform(0., radius_max)

        x_inside_batch = x_inside + x_perturb
        gradient_batch = np.empty(x_inside_batch.shape, dtype=np.float32)

        gradients = (self.substitute_model.batch_gradients(x_inside_batch * 255., [label] * n_samples) / 255.)
        if is_targeted:
            gradients = -gradients

        for i in range(n_samples):
            # Project the gradients.
            dot = np.vdot(gradients[i], source_direction)
            projected_gradient = gradients[i] - dot * source_direction          # Project orthogonal to source direction
            projected_gradient /= np.linalg.norm(projected_gradient)            # Norm to length 1
            gradient_batch[i] = projected_gradient

        return gradient_batch

    def _eval_sample(self, x, x_orig_normed=None):
        # Round, then get label and distance.
        x_rounded = np.round(np.clip(x * 255., 0, 255))
        preds = self.blackbox_model.predictions(np.uint8(x_rounded))
        label = np.argmax(preds)

        if x_orig_normed is None:
            return label
        else:
            dist = self.dm_main.calc(x_rounded/255., x_orig_normed)
            return label, dist
