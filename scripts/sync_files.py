import os
from absl import app, flags
from absl.flags import FLAGS
import shutil
from paramiko import SSHClient
from scp import SCPClient
import tqdm
import numpy as np
import copy

def move_files(name_file, destination_dir):

    print(f'Moving file {name_file}')
    current_path = os.path.join(os.getcwd(), name_file)
    destination_path = os.path.join(destination_dir, name_file)
    shutil.move(current_path, destination_path)
    print(f'file_saved_at {destination_dir} ')
    pass


def compress_files(dir_name, destination_dir=os.path.join(os.getcwd())):
    print(f'Compressing: {dir_name} ... ')
    output_name = os.path.split(dir_name)[-1]
    shutil.make_archive(output_name, 'zip', dir_name)
    move_files(output_name + '.zip', destination_dir)


def read_txt_file(dir_file):
    with open(dir_file, 'r') as f:
        lines = f.readlines()
    lines = [f.strip() for f in lines]
    f.close()
    return lines


def check_file_exists(path_list, name_file='list_experiments.txt'):
    list_files = os.listdir(path_list)
    if name_file in list_files:
        list_files = read_txt_file(os.path.join(path_list, name_file))
        return list_files
    else:
        return None


def transfer_files(local_path_results, name_file_local, name_file_remote):

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect('example.com')

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    scp.put('test.txt', 'test2.txt')
    scp.get('test2.txt')

    # Uploading the 'test' directory with its content in the
    # '/home/user/dump' remote directory
    scp.put('test', recursive=True, remote_path='/home/user/dump')

    scp.close()

    for file in list_missing:
        pass

    pass


def main(_argv):
    mode = FLAGS.mode
    remote = FLAGS.remote
    local = FLAGS.local
    local_path_results = FLAGS.local_path_results
    name_file_remote = FLAGS.name_file_remote
    name_file_local = FLAGS.name_file_local
    folder_results = FLAGS.folder_results
    transfer_results = FLAGS.transfer_results
    base_path = FLAGS.base_path
    output_file_name = FLAGS.output_file_name

    if mode == 'prepare_files':
        if not folder_results:
            raise ValueError('folder_results not defined')

        if not transfer_results:
            raise ValueError('transfer_results not defined')

        # read list experiment folders
        list_finished_experiments = [f for f in os.listdir(folder_results)
                                     if os.path.isdir(os.path.join(folder_results, f))]
        # read list .zip folders
        list_zipped_folders = [f for f in os.listdir(transfer_results) if f.endswith('.zip')]

        for i, experiment_folder in enumerate(tqdm.tqdm(list_finished_experiments, desc='Preparing files')):
            if experiment_folder+'.zip' not in list_zipped_folders:
                path_folder = os.path.join(folder_results, experiment_folder)
                compress_files(path_folder, destination_dir=transfer_results)

    elif mode == 'update_experiment_list':

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


    elif mode == 'transfer_results':
        list_files = os.listdir(local_path_results)

        if name_file_local in list_files:
            list_files_local = read_txt_file(os.path.join(local_path_results, name_file_local))
            list_files_remote = read_txt_file(os.path.join(local_path_results, name_file_remote))
            list_missing = [f for f in list_files_remote if f not in list_files_local]
            print(f'{len(list_missing)} need to be sync')


if __name__ == '__main__':

    flags.DEFINE_string('mode', 'prepare_files', 'prepare_files or transfer_files, update_experiment_list')
    flags.DEFINE_string('remote', None, 'dir to remote')
    flags.DEFINE_string('local', None, 'dir to local')
    flags.DEFINE_string('local_path_results', os.path.join(os.getcwd(), 'results'), 'dir to local results')
    flags.DEFINE_string('name_file_local', 'list_experiments.txt', 'dir to local results')
    flags.DEFINE_string('name_file_remote', 'list_remote_experiments.txt', 'dir to local results')
    flags.DEFINE_string('folder_results', None, 'folder with experiment folder')
    flags.DEFINE_string('transfer_results', None, 'folder with transfer folder')
    flags.DEFINE_string('base_path', None, 'base directory')
    flags.DEFINE_string('output_file_name', 'list_experiments.txt', 'name of the output file')

    try:
        app.run(main)
    except SystemExit:
        pass