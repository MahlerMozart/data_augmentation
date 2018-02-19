import math as m
import tensorflow as tf


def get_flipped_example(image, labels):
    """Mirror the inputs in a vertically centered line.

    Parameters
    ----------
    image: tensor
        3D tensor containing the image input
    labels: list
        list of 3D tensors containg different kind of tensors

    Returns
    -------
        Flipped input image and labels

    """
    flipped_image = tf.image.flip_left_right(image)
    flipped_labels = list()
    for label in labels:
        flipped_labels.append(tf.image.flip_left_right(label))
    return flipped_image, flipped_labels


def get_rotated_example(image, labels):
    """Rotate the inputs a random amount of degrees.

    A random amount of rotation is computed and applied to all the inputs. Then the data is croped
    to remove the black border introduced by the rotation. Finally the data is resized back to
    input size.

    Parameters
    ----------
    image: tensor
        3D tensor containing the image input
    labels: list
        list of 3D tensors containg different kind of tensors

    Returns
    -------
        Rotated image and labels

    """
    assert(image.get_shape().as_list()[0] == image.get_shape().as_list()[1])
    # Sample rotation angle in radians
    rotation_angle = tf.random_uniform([1], minval=-10, maxval=10)
    rotation_angle = tf.multiply(rotation_angle, tf.div(m.pi, 180))
    # Rotate image and label
    rotated_image = tf.contrib.image.rotate(image, rotation_angle, interpolation='BILINEAR')
    rotated_labels = list()
    for label in labels:
        rotated_labels.append(tf.contrib.image.rotate(
            label, rotation_angle, interpolation='NEAREST'))
    # Get the first row and column in the image
    first_row_of_image = tf.cast(rotated_image[0, :, :], tf.float32)
    first_column_of_image = tf.cast(rotated_image[:, 0, :], tf.float32)
    # Setup a condition that checks wether a pixel is zero or not
    condition = tf.constant([0, 0, 0], dtype=tf.float32)
    # Get a truth table for the column and row tensor
    row_condition_table = tf.not_equal(first_row_of_image, condition)
    # Get the indices for the black pixels
    row_zeros_indices = tf.where(row_condition_table)
    column_condition_table = tf.not_equal(first_column_of_image, condition)
    column_zeros_indices = tf.where(column_condition_table)
    # Since the height and width is the same we can just take the minimum of the indices to get
    # the thickness of the border
    border_width = tf.cast(tf.minimum(
        row_zeros_indices[0], column_zeros_indices[0]), tf.float32)
    # Get the input width
    width = tf.cast(tf.shape(image)[1], tf.float32)
    # Size after cropping is width subtracted by 2 times the border thickness
    crop_size = tf.subtract(width, tf.multiply(2.0, border_width[0]))
    crop_size = tf.cast(crop_size, tf.int32)
    out_size = tf.cast(width, tf.int32)
    # Crop out central part to remove padding of zeros from rotation and resize image back to
    # input size
    croped_image = tf.image.resize_image_with_crop_or_pad(rotated_image, crop_size, crop_size)
    croped_resized_image = tf.image.resize_images(croped_image, (out_size, out_size))
    # Crop and resize labels
    croped_resized_labels = list()
    for rotated_label in rotated_labels:
        croped_label = tf.image.resize_image_with_crop_or_pad(rotated_label, crop_size, crop_size)
        croped_resized_labels.append(tf.image.resize_images(
            croped_label, (out_size, out_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    return croped_resized_image, croped_resized_labels


def get_gaussian_noise_example(image, label):
    """Add gaussian noise to the image.

    Parameters
    ----------
    image: tensor
        3D tensor containing the image input
    labels: list
        list of 3D tensors containg different kind of tensors

    Returns
    -------
        Noisy image and labels

    """
    std = tf.multiply(0.15, 255.)
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=std, dtype=tf.float32)
    noisy_image = tf.add(image, noise)
    return noisy_image, label


def get_rescaled_example(image, labels):
    """Scale up the input data spatial shape and then crop to input shape.

    Applys a upscaling of the inputs spatial size followed by a crop to restore the input shape

    Parameters
    ----------
    image: tensor
        3D tensor containing the image input
    labels: list
        list of 3D tensors containg different kind of tensors

    Returns
    -------
        Zoomed image and labels

    """
    shape = tf.shape(image)
    resize_factor = tf.random_uniform([1], minval=1, maxval=1.6)
    resized_height = tf.multiply(tf.cast(shape[0], tf.float32), resize_factor[0])
    resized_width = tf.multiply(tf.cast(shape[1], tf.float32), resize_factor[0])
    # Cast dimensions
    resized_height = tf.cast(resized_height, tf.int32)
    resized_width = tf.cast(resized_width, tf.int32)
    height = tf.cast(shape[0], tf.int32)
    width = tf.cast(shape[1], tf.int32)
    # Resize and crop image and labels
    resized_image = tf.image.resize_images(image, (resized_height, resized_width))
    cropped_image = tf.image.resize_image_with_crop_or_pad(resized_image, height, width)
    # Crop and resize labels
    cropped_resized_labels = list()
    for label in labels:
        resized_label = tf.image.resize_images(
            label, (resized_height, resized_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        cropped_resized_labels.append(
            tf.image.resize_image_with_crop_or_pad(resized_label, height, width))
    return cropped_image, cropped_resized_labels
