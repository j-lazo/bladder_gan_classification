import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import numpy as np
import datetime
import tensorflow_addons as tfa


def generate_experiment_ID(name_model='', learning_rate='na', batch_size='na', backbone_model='',
                           prediction_model='', mode=''):
    """
    Generate a ID name for the experiment considering the name of the model, the learning rate,
    the batch size, and the date of the experiment

    :param name_model: (str)
    :param learning_rate: (float)
    :param batch_size: (int)
    :param backbone_model: (str)
    :return: (str) id name
    """
    if type(learning_rate) == list:
        lr = learning_rate[0]
    else:
        lr = learning_rate

    if prediction_model == '':
        training_date_time = datetime.datetime.now()
        if backbone_model != '':
            name_mod = ''.join([name_model, '+', backbone_model])
        else:
            name_mod = name_model
        id_name = ''.join([name_mod, '_', mode, '_lr_', str(lr),
                                  '_bs_', str(batch_size), '_',
                                  training_date_time.strftime("%d_%m_%Y_%H_%M")
                                  ])
    else:
        predictions_date_time = datetime.datetime.now()
        id_name = ''.join([prediction_model, '_predictions_', predictions_date_time.strftime("%d_%m_%Y_%H_%M")])

    return id_name


def load_data_from_directory(path_data, csv_annotations=None):
    """
        Give a path, creates two lists with the
        Parameters
        ----------
        path_data :

        Returns
        -------

        """

    dictionary_labels = {}
    list_classes = list()
    list_domain = list()
    list_imgs = list()
    list_files = list()
    list_path_files = list()

    csv_list = [f for f in os.listdir(path_data) if f.endswith('.csv')]

    if csv_annotations:
      data_frame = pd.read_csv(csv_annotations)
      list_imgs = data_frame['image_name'].tolist()
      list_classes = data_frame['tissue type'].tolist()
      list_domain = data_frame['imaging type'].tolist()

    else:
        if csv_list:
            csv_in_dir = csv_list.pop()
            print(f'csv file {csv_in_dir} found in {path_data}, using it as ground truth data')
            csv_annotations = os.path.join(path_data, csv_in_dir)
            data_frame = pd.read_csv(csv_annotations)
            list_imgs = data_frame['image_name'].tolist()
            list_classes = data_frame['tissue type'].tolist()
            list_domain = data_frame['imaging type'].tolist()

    list_unique_classes = np.unique(list_classes)
    list_unique_domains = np.unique(list_domain)

    for (dirpath, dirnames, filenames) in os.walk(path_data):
        list_files += [file for file in filenames]
        list_path_files += [os.path.join(dirpath, file) for file in filenames]

    # remove all the files which are not images
    list_files = [f for f in list_files if f.endswith('.png')]
    list_path_files = [f for f in list_path_files if f.endswith('.png')]

    for j, file in enumerate(list_files):
        if file in list_imgs:
            index_file = list_imgs.index(file)
            # here you have to find the index first
            new_dictionary_labels = {file: {'image_name': file, 'path_file': list_path_files[j],
                                            'img_class': list_classes[index_file],
                                            'img_domain': list_domain[index_file]}}
            dictionary_labels = {**dictionary_labels, **new_dictionary_labels}
        else:
            list_files.remove(file)

    print(f'Found {len(list_path_files)} images corresponding to {len(list_unique_classes)} classes and '
          f'{len(list_unique_domains)} domains at: {path_data}')

    return list_files, dictionary_labels


def make_tf_dataset(path, batch_size, training=False):

    global num_classes
    global training_mode
    training_mode = training

    def rand_degree(lower, upper):
        return random.uniform(lower, upper)

    def rotate_img(img, lower=0, upper=180):

        upper = upper * (np.pi / 180.0)  # degrees -> radian
        lower = lower * (np.pi / 180.0)
        img = tfa.image.rotate(img, rand_degree(lower, upper))
        return img

    def parse_image(filename):

        image = tf.io.read_file(filename)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [256, 256])

        if training_mode:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = rotate_img(image)

        return image

    def parse_label_output(y):

        def _parse(y):
            out_y = np.zeros(num_classes)
            out_y[y] = 1.
            out_y[y] = out_y[y].astype(np.float64)
            return out_y
        y_out = tf.numpy_function(_parse, [y], [tf.float64])[0]
        return y_out

    def configure_for_performance(dataset):
      dataset = dataset.shuffle(buffer_size=1000)
      dataset = dataset.batch(batch_size)
      dataset = dataset.repeat()
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset

    path_imgs = list()
    images_class = list()
    images_domains = list()
    csv_annotations_file = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')].pop()
    list_files, dictionary_labels = load_data_from_directory(path, csv_annotations=csv_annotations_file)
    if training:
        random.shuffle(list_files)

    for img_name in list_files:
      path_imgs.append(dictionary_labels[img_name]['path_file'])
      images_class.append(dictionary_labels[img_name]['img_class'])
      images_domains.append(dictionary_labels[img_name]['img_domain'])

    path_imgs = [f for f in path_imgs if f.endswith('.png')]
    list_path_files = path_imgs

    unique_domains = list(np.unique(images_domains))
    unique_classes = list(np.unique(images_class))
    num_classes = len(unique_classes)
    images_domains = [unique_domains.index(val) for val in images_domains]
    labels = [unique_classes.index(val) for val in images_class]
    network_labels = list()
    for label in labels:
        l = np.zeros(num_classes)
        l[label] = 1.0
        network_labels.append(l)

    filenames_ds = tf.data.Dataset.from_tensor_slices(list_path_files)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(network_labels)
    domain_ds = tf.data.Dataset.from_tensor_slices(images_domains)
    ds = tf.data.Dataset.zip(((images_ds, domain_ds), labels_ds))
    if training:
        ds = configure_for_performance(ds)
    else:
        ds = ds.batch(batch_size)

    return ds


def read_image_and_domain(dictionary_info, crop_size=256):
    domain_types = ['NBI', 'WLI']

    def imread(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, 3)
        return img

    def _map_fn(img, crop_size=32):  # preprocessing
        img = tf.image.resize(img, [crop_size, crop_size])
        # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
        img = tf.clip_by_value(img, 0, 255) / 255.0
        # or img = tl.minmax_norm(img)
        img = img * 2 - 1
        return img

    path_image = dictionary_info['path_file']
    image = imread(path_image)
    image = _map_fn(image, crop_size=crop_size)
    image_domain = domain_types.index(dictionary_info['img_domain'])
    domain = tf.constant([image_domain])

    return image, domain