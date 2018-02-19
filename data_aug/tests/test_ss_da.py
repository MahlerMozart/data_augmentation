import unittest
import numpy as np
import tensorflow as tf

from data_aug.semantic_segmentation.augment import\
    get_flipped_example, get_rotated_example, get_gaussian_noise_example, get_rescaled_example
from data_aug.semantic_segmentation.get_data import\
    get_example, get_augmented_data


class TestSSDA(unittest.TestCase):

    def setUp(self):

        self.input_img = np.random.normal(
            loc=0.0, scale=1.0, size=(512, 512, 3)).astype(np.float32)
        self.image_ph = tf.placeholder(tf.float32, [512, 512, 3])
        self.feed_dict = {self.image_ph: self.input_img}
        # Data augmentation selection ops
        self.get_ex = get_example(self.image_ph, [self.image_ph, self.image_ph])
        self.get_aug_data = get_augmented_data(self.image_ph, [self.image_ph, self.image_ph])
        # Data augmenation ops
        self.flip_ex = get_flipped_example(self.image_ph, [self.image_ph, self.image_ph])
        self.rot_ex = get_rotated_example(self.image_ph, [self.image_ph, self.image_ph])
        self.gauss_ex = get_gaussian_noise_example(self.image_ph, [self.image_ph, self.image_ph])
        self.resc_ex = get_rescaled_example(self.image_ph, [self.image_ph, self.image_ph])

    def run_tf_op(self, operation):

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(operation, self.feed_dict)

    def test_get_example(self):
        image, labels = self.run_tf_op(self.get_ex)
        assert(image.shape == self.input_img.shape)

    def test_get_augmented_data(self):
        image, labels = self.run_tf_op(self.get_aug_data)
        assert(image.shape == self.input_img.shape)

    def test_get_flipped_example(self):
        image, labels = self.run_tf_op(self.flip_ex)
        assert(image.shape == self.input_img.shape)
        assert((image != self.input_img).any())

    def test_get_rotated_example(self):
        image, labels = self.run_tf_op(self.rot_ex)
        assert(image.shape == self.input_img.shape)
        assert((image != self.input_img).any())

    def test_get_gaussian_noise_example(self):
        image, labels = self.run_tf_op(self.gauss_ex)
        assert(image.shape == self.input_img.shape)
        assert((image != self.input_img).any())
        assert((labels[0] == self.input_img).any())

    def test_get_rescaled_example(self):
        image, labels = self.run_tf_op(self.resc_ex)
        assert(image.shape == self.input_img.shape)
        assert((image != self.input_img).any())


if __name__ == '__main__':
    unittest.main()
