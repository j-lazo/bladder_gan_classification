import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import numpy as np
import datetime
import tensorflow_addons as tfa
from utils import image as img_fun
import copy


def generate_experiment_ID(name_model='', learning_rate='na', batch_size='na', backbone_model='',
                           prediction_model='', mode='', specific_domain=None):
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

        if specific_domain:
            id_name = ''.join([name_mod, '_', mode, '_lr_', str(lr),
                              '_bs_', str(batch_size), '_', 'trained_with_', specific_domain, '_',
                               training_date_time.strftime("%d_%m_%Y_%H_%M")
                               ])
        else:
            id_name = ''.join([name_mod, '_', mode, '_lr_', str(lr),
                              '_bs_', str(batch_size), '_',
                               training_date_time.strftime("%d_%m_%Y_%H_%M")
                               ])
    else:
        predictions_date_time = datetime.datetime.now()
        id_name = ''.join([prediction_model, '_predictions_', predictions_date_time.strftime("%d_%m_%Y_%H_%M")])

    return id_name


def select_gan_specific(list_files, input_dictionary, gan_selection):
    output_list_files = list()
    output_dict = {}

    gan_inclusion_dict = {'converted': '_converted',
                'reconverted': '_reconverted',
                'generated': 'converted'}

    gan_exclusion_dict = {'all+converted': '_reconverted',
                          'all+reconverted': '_converted',
                          'all+converted_NBI': '_reconverted'}
    if 'NBI' in gan_selection:
        specific_domain = 'NBI'
    else:
        specific_domain = None

    for file in list_files:
        index_file = list_files.index(file)
        if gan_selection in gan_inclusion_dict:
            if gan_inclusion_dict[gan_selection] in file:
                output_list_files.append(file)
                output_dict[file] = input_dictionary[file]

        if gan_selection in gan_exclusion_dict:
            if gan_exclusion_dict[gan_selection] not in file:
                if specific_domain:
                    if '_converted' in file and input_dictionary[file] == specific_domain:
                        output_list_files.append(file)
                        output_dict[file] = input_dictionary[file]
                    else:
                        output_list_files.append(file)
                        output_dict[file] = input_dictionary[file]
                else:
                    output_list_files.append(file)
                    output_dict[file] = input_dictionary[file]

    print(output_list_files)
    print(output_dict)
    return output_list_files, output_dict


def load_data_from_directory(path_data, csv_annotations=None, specific_domain=None, case_specific=None,
                             gan_selection=None):
    """
        Give a path, creates two lists with the
        Parameters
        ----------
        path_data :

        Returns
        -------

        """

    dictionary_labels = {}
    list_files = list()
    list_path_files = list()

    csv_list = [f for f in os.listdir(path_data) if f.endswith('.csv')]

    if csv_annotations:
      data_frame = pd.read_csv(csv_annotations)

    else:
        if csv_list:
            csv_in_dir = csv_list.pop()
            print(f'csv file {csv_in_dir} found in {path_data}, using it as ground truth data')
            csv_annotations = os.path.join(path_data, csv_in_dir)
            data_frame = pd.read_csv(csv_annotations)

    list_imgs = data_frame['image_name'].tolist()
    list_classes = data_frame['tissue type'].tolist()
    list_domain = data_frame['imaging type'].tolist()


    # choose case specific files
    if case_specific:
        list_cases = data_frame['case number'].tolist()
        list_imgs = [image for i, image in enumerate(list_imgs) if list_cases[i] in case_specific]

    list_unique_classes = list(np.unique(list_classes))
    if 'nan' in list_unique_classes:
        list_unique_classes.remove('nan')

    list_unique_domains = np.unique(list_domain)

    for (dirpath, dirnames, filenames) in os.walk(path_data):
        list_files += [file for file in filenames]
        list_path_files += [os.path.join(dirpath, file) for file in filenames]

    # remove all the files which are not images
    list_files = [f for f in list_files if f.endswith('.png')]
    list_path_files = [f for f in list_path_files if f.endswith('.png')]
    output_list_files = list()

    print(f'Found {len(list_path_files)} images corresponding to {len(list_unique_classes)} classes and '
          f'{len(list_unique_domains)} domains at: {path_data}')

    for j, file in enumerate(list_files):
        if file in list_imgs:
            # find the index first
            index_file = list_imgs.index(file)
            if specific_domain:
                if list_domain[index_file] == specific_domain:
                    new_dictionary_labels = {file: {'image_name': file, 'path_file': list_path_files[j],
                                                    'img_class': list_classes[index_file],
                                                    'img_domain': list_domain[index_file]}}
                    output_list_files.append(file)
                    dictionary_labels = {**dictionary_labels, **new_dictionary_labels}

            else:
                new_dictionary_labels = {file: {'image_name': file, 'path_file': list_path_files[j],
                                                'img_class': list_classes[index_file],
                                                'img_domain': list_domain[index_file]}}
                output_list_files.append(file)
                dictionary_labels = {**dictionary_labels, **new_dictionary_labels}
        else:
            list_files.remove(file)

    if gan_selection:
        output_list_files, dictionary_labels = select_gan_specific(output_list_files, dictionary_labels, gan_selection)

    if specific_domain:
        print(f'Only images from domain {specific_domain} selected, in total {len(output_list_files)} images '
              f'are considered in the dataset')

    return output_list_files, dictionary_labels


def make_tf_dataset(list_files, dictionary_labels, batch_size, training=False, multi_output=False, specific_domain=None,
                    num_repeat=None, custom_training=False, ignore_labels=False):

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

        image = img_fun.decode_image(filename)
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
      if num_repeat:
          dataset = dataset.repeat()
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset

    path_imgs = list()
    images_class = list()
    images_domains = list()

    if training:
        random.shuffle(list_files)

    for img_name in list_files:
      path_imgs.append(dictionary_labels[img_name]['path_file'])
      images_class.append(dictionary_labels[img_name]['img_class'])
      images_domains.append(dictionary_labels[img_name]['img_domain'])

    path_imgs = [f for f in path_imgs if f.endswith('.png')]
    unique_domains = list(np.unique(images_domains))
    images_domains = [unique_domains.index(val) for val in images_domains]

    # here you need to make a selection of the specific domains, but once both have been considered
    if specific_domain:
        new_path_imgs = list()
        new_images_class = list()
        new_images_domains = list()
        for j, img in enumerate(path_imgs):
            if unique_domains.index(specific_domain) == images_domains[j]:
                new_path_imgs.append(path_imgs[j])
                new_images_class.append(images_class[j])
                new_images_domains.append(images_domains[j])

        path_imgs = copy.copy(new_path_imgs)
        images_class = copy.copy(new_images_class)
        images_domains = copy.copy(new_images_domains)

    filenames_ds = tf.data.Dataset.from_tensor_slices(path_imgs)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if ignore_labels:
        num_classes = np.nan
        domain_ds = tf.data.Dataset.from_tensor_slices(images_domains)
        ds = tf.data.Dataset.zip((images_ds, domain_ds))

    else:
        unique_classes = list(np.unique(images_class))
        num_classes = len(unique_classes)
        labels = [unique_classes.index(val) for val in images_class]
        network_labels = list()

        if custom_training:
            labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        else:
            # create a vector according to the label where 1 is the position corresponding to the value
            for label in labels:
                l = np.zeros(num_classes)
                l[label] = 1.0
                network_labels.append(l)
            labels_ds = tf.data.Dataset.from_tensor_slices(network_labels)

        if multi_output:
            domain_ds = tf.data.Dataset.from_tensor_slices(images_domains)
            ds = tf.data.Dataset.zip(((images_ds, domain_ds), labels_ds))
        else:
            ds = tf.data.Dataset.zip((images_ds, labels_ds))

    if training:
        ds = configure_for_performance(ds)
    else:
        ds = ds.batch(batch_size)

    print(f'TF dataset of {len(path_imgs)} elements')
    return ds, num_classes


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