import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import numpy as np
import datetime


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

    list_files = os.listdir(path_data)
    csv_list = [f for f in list_files if f.endswith('.csv')]
    if csv_list:
      csv_indir = csv_list.pop()

    if csv_annotations:
      data_frame = pd.read_csv(csv_annotations)
      list_imgs = data_frame['image_name'].tolist()
      list_classes = data_frame['tissue type'].tolist()
      list_domain = data_frame['imaging type'].tolist()

    list_unique_classes = np.unique(list_classes)
    list_unique_domains = np.unique(list_domain)
    list_files = list()
    list_path_files = list()

    for (dirpath, dirnames, filenames) in os.walk(path_data):
      list_files += [file for file in filenames]
      list_path_files += [os.path.join(dirpath, file) for file in filenames]

    list_files = [f for f in list_files if f.endswith('.png')]
    for j, file in enumerate(list_imgs):
      if file in list_files:
        new_dictionary_labels = {file: {'image_name': file, 'path_file': list_path_files[j],
                                        'img_class': list_classes[j], 'img_domain': list_domain[j]}}
        dictionary_labels = {**dictionary_labels, **new_dictionary_labels}
      else:
        list_files.remove(file)

    print(f'Found {len(list_path_files)} images corresponding to {len(list_unique_classes)} classes and '
          f'{len(list_unique_domains)} domains at: {path_data}')

    return list_files, dictionary_labels


def make_tf_dataset(path, batch_size):

    global num_classes

    def parse_image(filename):
      image = tf.io.read_file(filename)
      image = tf.image.decode_png(image, channels=3)
      image = tf.image.resize(image, [256, 256])
      return image

    def parse_label_output(y):

        def _parse(y):
            out_y = np.zeros(num_classes)
            out_y[y] = 1.
            out_y[y] = out_y[y].astype(np.float64)
            return out_y
        y_out = tf.numpy_function(_parse, [y], [tf.float64])[0]
        return y_out

    def parse_binary_label(y):
        out_y = np.zeros(1)
        out_y[y] = np.float32(y)
        out_y[y] = out_y[y].astype(np.float64)
        return out_y

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
    # = labels_ds.map(parse_label_output, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    domain_ds = tf.data.Dataset.from_tensor_slices(images_domains)
    ds = tf.data.Dataset.zip(((images_ds, domain_ds), labels_ds))
    ds = configure_for_performance(ds)

    return ds