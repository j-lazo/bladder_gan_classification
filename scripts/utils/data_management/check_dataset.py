from absl import app, flags
from absl.flags import FLAGS
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import os
import datasets as ds
import numpy as np

def main(_argv):
    print(tf.__version__)
    path_dataset = FLAGS.path_dataset
    batch_size = FLAGS.batch_size
    img_width = FLAGS.img_width
    img_height = FLAGS.img_height

    classes = [f for f in os.listdir(path_dataset) if not f.endswith('.csv')]
    domains = ['NBI', 'WLI']
    print(classes)

    csv_files = [f for f in os.listdir(path_dataset) if f.endswith('.csv')]
    if csv_files:
        csv_file = csv_files.pop()
        csv_file_dir = path_dataset + csv_file
        df = pd.read_csv(csv_file_dir)

    dataset = ds.make_tf_dataset(path_dataset, batch_size)

    plt.figure()
    ncols = int(batch_size / 2)

    for inputs, label in dataset.take(5):
        image = inputs[0]
        domain = inputs[1]
        label = label.numpy()
        for i in range(batch_size):
            class_index = int(np.where(label[i] == 1.)[0][0])
            plt.subplot(2, ncols, i + 1)
            plt.gca().set_title(classes[class_index] + ' ' + domains[domain[i]])
            plt.imshow(image[i] / 255)
            plt.axis('off')

        plt.show()


if __name__ == '__main__':
    flags.DEFINE_string('path_dataset', '', 'path to the dataset directory')
    flags.DEFINE_integer('batch_size', 32, 'batch size')
    flags.DEFINE_integer('img_width', 256, 'batch size')
    flags.DEFINE_integer('img_height', 256, 'batch size')
    try:
        app.run(main)
    except SystemExit:
        pass