import tensorflow as tf


def decode_image(file_name):

    image = tf.io.read_file(file_name)
    if file_name.endswith('.png'):
        image = tf.image.decode_png(image, channels=3)
    elif file_name.endswith('.jpg'):
        image = tf.io.decode_jpeg(image, channels=3)
    elif file_name.endswith('.jpeg'):
        image = tf.io.decode_jpeg(image, channels=3)
    elif file_name.endswith('.gif'):
        image = tf.io.decode_gif(image)
    elif file_name.endswith('.bmp'):
        image = tf.io.decode_bmp(image, channels=3)
    else:
        raise TypeError(f' Not able to decode {file_name}. File type {file_name[-4:]} not in image options')

    return image