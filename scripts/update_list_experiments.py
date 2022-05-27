import copy
import subprocess
import os
from absl import app, flags
from absl.flags import FLAGS
import numpy as np


def read_file(dir_file):
    with open(dir_file, 'r') as f:
        lines = f.readlines()
    lines = [f.strip() for f in lines]
    f.close()
    return lines


def check_file_exists(path_list, name_file='list_experiments.txt'):
    list_files = os.listdir(path_list)
    if name_file in list_files:
        list_files = read_file(os.path.join(path_list, name_file))
        return list_files
    else:
        return None


def update_experiment_list(_argv):
    base_path = FLAGS.base_path
    output_file_name = FLAGS.output_file_name

    if base_path:
        directory_path = os.path.join(base_path, 'results', 'bladder_tissue_classification_v2')
    else:
        directory_path = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')

    list_files = [f for f in os.listdir(directory_path)
                  if os.path.isdir(os.path.join(directory_path, f)) or f.endswith('.zip')]
    # remove repeated zip a and/or dir files
    unique_files = np.unique([f.replace('.zip', '') for f in list_files])
    list_existing_files = check_file_exists(directory_path, name_file=output_file_name)
    list_files_update = list(copy.copy(unique_files))
    for f in unique_files:
        if list_existing_files:
            if f not in list_existing_files:
                list_files_update.append(f)

    f = open(os.path.join(directory_path, output_file_name), 'w')
    for i, experiment_file in enumerate(list_files_update):
        f.write(''.join([experiment_file, "\r\n"]))

    f.close()
    print(f'list experiments file updated at dir: {directory_path}')


if __name__ == '__main__':

    flags.DEFINE_string('base_path', None, 'base directory')
    flags.DEFINE_string('output_file_name', 'list_experiments.txt', 'name of the output file')
    try:
        app.run(update_experiment_list)
    except SystemExit:
        pass

