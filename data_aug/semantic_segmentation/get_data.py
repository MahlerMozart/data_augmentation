import tensorflow as tf
from .augment import get_flipped_example, get_rotated_example, get_gaussian_noise_example,\
    get_rescaled_image


def get_example(image, labels, prob=0.5):
    """Provide augmented or original input data with probablitity according to input parameter.

    Parameters
    ----------
    image: tensor
        raw input image
    labels: list
        labels
    prob: int
        probablity that the returned data is augmented by a random data augmentation function

    Returns
    -------
        Data with random augmentation or original input data

    """
    augment = tf.random_uniform([1], maxval=1, dtype=tf.float32)[0]
    image_aug, labels_aug = tf.cond(augment < prob,
                                    lambda: get_augmented_data(image, labels),
                                    lambda: (image, labels))
    return image_aug, labels_aug


def get_augmented_data(image, labels):
    """Provide data with random augmentation.

    Parameters
    ----------
    image: tensor
        raw input image
    labels: list
        labels

    Returns
    -------
        Data with random augmentation

    """
    augmentation_index = tf.random_uniform([1], maxval=4, dtype=tf.int32)[0]

    def aug1(image, labels): return get_flipped_example(image, labels)

    def aug2(image, labels): return get_rotated_example(image, labels)

    def aug3(image, labels): return get_gaussian_noise_example(image, labels)

    def aug4(image, labels): return get_rescaled_image(image, labels)

    image, labels = tf.case([(tf.equal(augmentation_index, 0), lambda: aug1(image, labels)),
                             (tf.equal(augmentation_index, 1), lambda: aug2(image, labels)),
                             (tf.equal(augmentation_index, 2), lambda: aug3(image, labels)),
                             (tf.equal(augmentation_index, 3), lambda: aug4(image, labels))],
                            default=lambda: aug1(image, labels))
    return image, labels
