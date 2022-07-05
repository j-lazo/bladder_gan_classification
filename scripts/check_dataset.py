from absl import app, flags
from absl.flags import FLAGS
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import os
from utils.data_management import datasets as ds
import numpy as np
from utils.data_management import datasets as dam
import tqdm


def main(_argv):
    print(tf.__version__)
    path_dataset = os.path.normpath(FLAGS.path_dataset)
    batch_size = FLAGS.batch_size
    img_width = FLAGS.img_width
    img_height = FLAGS.img_height
    training = FLAGS.training
    specific_domain = FLAGS.specific_domain

    classes = [f for f in os.listdir(path_dataset) if not f.endswith('.csv')]
    domains = ['NBI', 'WLI']
    print(classes)

    csv_files = [f for f in os.listdir(path_dataset) if f.endswith('.csv')]
    if csv_files:
        csv_file = csv_files.pop()
        csv_file_dir = os.path.join(path_dataset, csv_file)
        df = pd.read_csv(csv_file_dir)

    list_data, dictionary_data = dam.load_data_from_directory(path_dataset, csv_annotations=csv_file_dir,
                                                             specific_domain=specific_domain)
    dataset = dam.make_tf_dataset(list_data, dictionary_data, batch_size, training=True, multi_output=True,
                                  specific_domain='NBI')

    plt.figure()
    ncols = int(batch_size / 2)
    for inputs, labels in dataset.take(5):
        image = inputs[0]
        domain = inputs[1]
        labels = labels.numpy()
        for i in range(batch_size):
            class_index = int(np.where(labels[i] == 1.)[0][0])
            plt.subplot(2, ncols, i + 1)
            plt.gca().set_title(classes[class_index] + ' ' + domains[domain[i]])
            plt.imshow(image[i] / 255)
            plt.axis('off')

        plt.show()


if __name__ == '__main__':
    flags.DEFINE_string('path_dataset', '', 'path to the dataset directory')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('img_width', 256, 'batch size')
    flags.DEFINE_integer('img_height', 256, 'batch size')
    flags.DEFINE_bool('training', False, 'Take training parameters or not to create the Data')
    flags.DEFINE_string('specific_domain', '', 'in case: WLI/NBI')
    try:
        app.run(main)
    except SystemExit:
        pass