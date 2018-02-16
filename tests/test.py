from PIL import Image
import numpy as np
import tensorflow as tf


def test():

    filename_queue = tf.train.string_input_producer(
        ['/home/jonas/Repositories/data_augmentation/tests/garden.jpeg'])
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    img = tf.image.decode_jpeg(value)
    img = tf.image.resize_images(img, [512, 512])
    image, labels = get_example(img, [img, img])
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image_, labels_ = sess.run([image, labels])
        image_formated = np.squeeze(np.asarray(image_, dtype=np.uint8))
        Image.fromarray(image_formated).show()
        for label in labels_:
            label_formated = np.squeeze(np.asarray(label, dtype=np.uint8))
            Image.fromarray(label_formated).show()
        coord.request_stop()
        coord.join(threads)
