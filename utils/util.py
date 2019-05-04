import numpy as np


def line_search_to_boundary(bb_model, x_orig, x_start, label, is_targeted):
    """
    Binary search along a line between start and original image in order to find the decision boundary.
    :param bb_model: The (black-box) model.
    :param x_orig: The original image to attack.
    :param x_start: The starting image (which fulfills the adversarial criterion)
    :param is_targeted: true if the attack is targeted.
    :param label: the target label if targeted, or the correct label if untargeted.
    :return: A point next to the decision boundary (but still adversarial)
    """

    eps = 0.5  # Stop when decision boundary is closer than this (in L2 distance)
    i = 0

    x1 = np.float32(x_start)
    x2 = np.float32(x_orig)

    diff = x2 - x1
    while np.linalg.norm(diff / 255.) > eps:
        i += 1

        x_candidate = x1 + 0.5 * diff
        if (np.argmax(bb_model.predictions(x_candidate)) == label) == is_targeted:
            x1 = x_candidate
        else:
            x2 = x_candidate

        diff = x2 - x1

    print("Found decision boundary after {} queries.".format(i))
    return x1


def find_closest_img(bb_model, X_orig, X_targets, label, is_targeted):
    """
    From a list of potential starting images, finds the closest to the original.
    Before returning, this method makes sure that the image fulfills the adversarial condition (is actually classified as the target label).
    :param bb_model: The (black-box) model.
    :param X_orig: The original image to attack.
    :param X_targets: List of images that fulfill the adversarial criterion (i.e. target class in the targeted case)
    :param is_targeted: true if the attack is targeted.
    :param label: the target label if targeted, or the correct label if untargeted.
    :return: the closest image (in L2 distance) to the original that also fulfills the adversarial condition.
    """

    X_orig_normed = np.float32(X_orig) / 255.
    dists = np.empty(len(X_targets), dtype=np.float32)
    for i in range(len(X_targets)):
        d_l2 = np.linalg.norm((np.float32(X_targets[i, ...]) / 255. - X_orig_normed))
        dists[i] = d_l2

    indices = np.argsort(dists)
    for index in indices:
        X_target = X_targets[index]
        pred_clsid = np.argmax(bb_model.predictions(X_target))
        if (pred_clsid == label) == is_targeted:
            print("Found an image of the target class, d_l2={:.3f}.".format(dists[index]))
            return X_target

        print("Image of target class is wrongly classified by model, skipping.")

    raise ValueError("Could not find an image of the target class that was correctly classified by the model!")


def sample_hypersphere(n_samples, sample_shape, radius, sample_gen=None, seed=None):
    """
    Uniformly sample the surface of a L2-hypersphere.
    Uniform picking: create a n-dimensional normal distribution and then normalize it to the desired radius.
    See http://mathworld.wolfram.com/HyperspherePointPicking.html
    :param n_samples: number of image samples to generate.
    :param sample_shape: shape of a single image sample.
    :param radius: radius(=eps) of the hypersphere.
    :param sample_gen: If provided, retrieves random numbers from this generator.
    :param seed: seed for the random generator. Cannot be used with the sample generator.
    :return: Batch of image samples, shape: (n_samples,) + sample_shape
    """

    if sample_gen is not None:
        assert seed is None, "Can't provide individual seeds if using the multi-threaded generator."
        assert sample_shape == sample_gen.shape

        # Get precalculated samples from the generator
        gauss = np.empty(shape=(n_samples, np.prod(sample_shape)), dtype=np.float32)
        for i in range(n_samples):
            gauss[i] = sample_gen.get_normal().reshape(-1)
    else:
        if seed is not None:
            np.random.seed(seed)
        gauss = np.random.normal(size=(n_samples, np.prod(sample_shape)))

    # Norm to 1
    norm = np.linalg.norm(gauss, ord=2, axis=1)
    perturbation = (gauss / norm[:, np.newaxis]) * radius

    perturbation = np.reshape(perturbation, (n_samples,) + sample_shape)
    return perturbation
