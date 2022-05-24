import tensorflow as tf


def decode_image(file_name):

    image = tf.io.read_file(file_name)
    if tf.io.is_jpeg(image):
        image = tf.io.decode_jpeg(image, channels=3)
    else:
        image = tf.image.decode_png(image, channels=3)

    return image