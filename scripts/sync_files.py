import os

import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
import shutil
from paramiko import SSHClient
from scp import SCPClient
import tqdm
import numpy as np
import copy
import yaml
from matplotlib import pyplot as plt
import seaborn as sns


def analyze_individual_cases(name_csv_file):

    df = pd.read_csv(name_csv_file)

    fig1 = plt.figure(1, figsize=(11, 7))
    ax1 = sns.boxplot(x="name_model", y="acc all", data=df)
    ax1 = sns.swarmplot(x="name_model", y="acc all", data=df, color=".25")
    ax1.set_ylim([0.5, 1.0])
    ax1.title.set_text('')

    plt.show()


def read_yaml_file(path_file):
    with open(path_file) as file:
        document = yaml.full_load(file)

    return document
        #for item, doc in documents.items():
        #    print(item, ":", doc)


def move_files(name_file, destination_dir):

    print(f'Moving file {name_file}')
    current_path = os.path.join(os.getcwd(), name_file)
    destination_path = os.path.join(destination_dir, name_file)
    shutil.move(current_path, destination_path)
    print(f'file_saved_at {destination_dir} ')


def find_pattern_names(string_name, str_pattern):
    """
    Looks for a pattern name in a string and returns the number after it
    :param string_name: the string where to look for a pattern
    :param str_pattern: the pattern that needs to be found
    :return:
    """
    match = re.search(str_pattern + '(\d+)', string_name)
    if match:
        return match.group(1)
    else:
        return np.nan


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


def transfer_files(local_host):

    ssh = SSHClient()
    ssh.load_system_host_keys()
    print(f'connecting to {local_host}')
    ssh.connect(local_host)

    # SCPCLient takes a paramiko transport as an argument
    scp = SCPClient(ssh.get_transport())

    #scp.put('test.txt', 'test2.txt')
    #scp.get('test2.txt')
    scp.close()


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
    gt_results_file = FLAGS.gt_results_file
    output_csv_file = FLAGS.output_csv_file

    if mode == 'prepare_files':
        compressed_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_compressed')
        if not folder_results:
            folder_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')
        else:
            print(os.path.isdir(folder_results))

        if not transfer_results:
            transfer_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_transfer')
        else:
            print(os.path.isdir(transfer_results))

        # read list experiment folders
        list_finished_experiments = [f for f in os.listdir(folder_results)
                                     if os.path.isdir(os.path.join(folder_results, f))]

        # check the experiment is finished
        list_finished_experiments = [f for f in list_finished_experiments
                                     if len(os.listdir(os.path.join(folder_results, f))) >= 7]

        # read list .zip folders
        list_zipped_folders = [f for f in os.listdir(transfer_results) if f.endswith('.zip')]

        for i, experiment_folder in enumerate(tqdm.tqdm(list_finished_experiments, desc='Preparing files')):
            # compress the whole file including weights
            if experiment_folder+'.zip' not in list_zipped_folders:
                path_folder = os.path.join(folder_results, experiment_folder)
                compress_files(path_folder, destination_dir=compressed_results)
                # make a temporal folder with only the files you want to compress
                temporal_folder_dir = os.path.join(os.getcwd(), 'results', experiment_folder)
                os.mkdir(temporal_folder_dir)
                # move files to temporal folder
                list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                              f.endswith('.yaml') or f.endswith('.csv')]
                for file_name in list_files:
                    shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
                compress_files(temporal_folder_dir, destination_dir=transfer_results)
                # compress the file
                # delete the previous temporal one
                print(f'going to remove {temporal_folder_dir}')
                #try:
                #    shutil.rmtree(temporal_folder_dir)
                #except OSError as e:
                #    print("Error: %s - %s." % (e.filename, e.strerror))
            # compress only the files to transfer

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

    elif mode == 'unizp_and_analyze':

        # unzip the files that haven't been unzip
        dir_zipped_files = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_transfer')
        dir_unzipped_files = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')

        #df_real_values = pd.read_csv(gt_results_file)

        list_transfered_files = [f for f in os.listdir(dir_zipped_files) if f.endswith('.zip')]
        list_unzipped_files = [f for f in os.listdir(dir_unzipped_files) if os.path.isdir(os.path.join(dir_unzipped_files, f))]

        for i, zipped_file in enumerate(tqdm.tqdm(list_transfered_files, desc='Preparing files')):
            if zipped_file.replace('.zip', '') not in list_unzipped_files:
                filename = os.path.join(dir_zipped_files, zipped_file)
                dir_unzipped = os.path.join(dir_unzipped_files, zipped_file.replace('.zip', ''))
                #if not os.path.isdir(dir_unzipped):
                #    os.mkdir(dir_unzipped)
                #print(f'Expanding {zipped_file}  ...')
                #shutil.unpack_archive(filename, dir_unzipped)

        # analyze the yaml file

        name_experiment = list()
        date_experiment = list()
        name_models = list()
        backbones = list()

        dictionary_experiments = {}

        list_experiments = [f for f in os.listdir(dir_unzipped_files) if os.path.isdir(os.path.join(dir_unzipped_files, f))]
        for j, experiment_folder in enumerate(tqdm.tqdm(list_experiments, desc='Preparing files')):
            # extract the information from the folder's name

            name_model = experiment_folder.split('_fit')[0]
            ending = experiment_folder.split('lr_')[1]
            learning_rate = float(ending.split('_bs')[0])
            pre_bs = ending.split('bs_')[1]
            batch_size = int(pre_bs.split('_')[0])
            date = ending.split(''.join(['bs_', str(batch_size), '_']))[1]

            name_experiment.append(experiment_folder)
            date_experiment.append(date)
            name_models.append(name_model)
            #print(learning_rate)
            #learning_rates.append(learning_rate)

            if '+' in name_model:
                backbone_name = name_model.split('+')[1]
                backbones.append(backbone_name)
            else:
                backbone_name = ''
                backbones.append('')

            yaml_file = [f for f in os.listdir(os.path.join(dir_unzipped_files, experiment_folder)) if f.endswith('.yaml')]
            if yaml_file:
                yaml_file = yaml_file.pop()
                path_yaml = os.path.join(dir_unzipped_files, experiment_folder, yaml_file)
                results = read_yaml_file(path_yaml)
                acc_all = float(results['Accuracy ALL '])
                acc_nbi = float(results['Accuracy NBI '])
                acc_wli = float(results['Accuracy WLI '])

            else:
                acc_all = None
                acc_nbi = None
                acc_wli = None

            information_experiment = {'experiment_folder': experiment_folder, 'date': date, 'name_model': name_model,
                                      'backbones': backbone_name, 'batch_size': batch_size, 'learning_rate': learning_rate,
                                      'acc all': acc_all, 'acc wli': acc_wli, 'acc nbi': acc_nbi}

            dictionary_experiments.update({experiment_folder: information_experiment})

        print(np.unique(name_models))
        list_acc_all = [dictionary_experiments[experiment]['acc all'] for experiment in dictionary_experiments]
        list_acc_nbi = [dictionary_experiments[experiment]['acc nbi'] for experiment in dictionary_experiments]
        list_acc_wli = [dictionary_experiments[experiment]['acc wli'] for experiment in dictionary_experiments]

        sorted_list_all = [x for _, x in sorted(zip(list_acc_all, dictionary_experiments))]
        sorted_list_nbi = [x for _, x in sorted(zip(list_acc_nbi, dictionary_experiments))]
        sorted_list_wli = [x for _, x in sorted(zip(list_acc_wli, dictionary_experiments))]

        sorted_dict = dict()
        srted_nbi = dict()
        sorted_wli = dict()

        for x in reversed(sorted_list_all):
            sorted_dict.update({x: dictionary_experiments[x]})

        for x in reversed(sorted_list_nbi):
            srted_nbi.update({x: dictionary_experiments[x]})

        for x in reversed(sorted_list_wli):
            sorted_wli.update({x: dictionary_experiments[x]})

        new_df = pd.DataFrame.from_dict(dictionary_experiments, orient='index')
        output_csv_file_dir = os.path.join(os.getcwd(), 'results', output_csv_file)
        new_df.to_csv(output_csv_file_dir, index=False)

        sorted_df = pd.DataFrame.from_dict(sorted_dict, orient='index')
        output_csv_file_dir2 = os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')
        sorted_df.to_csv(output_csv_file_dir2, index=False)

        sorted_df_nbi = pd.DataFrame.from_dict(srted_nbi, orient='index')
        output_csv_file_dir3 = os.path.join(os.getcwd(), 'results', 'sorted_nbi_experiments_information.csv')
        sorted_df_nbi.to_csv(output_csv_file_dir3, index=False)

        sorted_df_wli = pd.DataFrame.from_dict(sorted_wli, orient='index')
        output_csv_file_dir4 = os.path.join(os.getcwd(), 'results', 'sorted_wli_experiments_information.csv')
        sorted_df_wli.to_csv(output_csv_file_dir4, index=False)
        analyze_individual_cases(output_csv_file_dir)

    elif mode == 'transfer_results':
        list_files = os.listdir(local_path_results)

        # first compare which files are missing in local
        list_missing = list()
        if name_file_local in list_files:
            list_files_local = read_txt_file(os.path.join(local_path_results, name_file_local))
            list_files_remote = read_txt_file(os.path.join(local_path_results, name_file_remote))
            list_missing = [f for f in list_files_remote if f not in list_files_local]
            print(f'{len(list_missing)} need to be sync')

        for i, missing_file in enumerate(tqdm.tqdm(list_missing, desc='Preparing files')):
            print(f'Coping {missing_file} to local')
            transfer_files(local)


if __name__ == '__main__':

    flags.DEFINE_string('mode', 'prepare_files', 'prepare_files or transfer_files or  update_experiment_list or unizp_and_analyze')
    flags.DEFINE_string('remote', None, 'dir to remote')
    flags.DEFINE_string('local', None, 'dir to local')
    flags.DEFINE_string('local_path_results', os.path.join(os.getcwd(), 'results'), 'dir to local results')
    flags.DEFINE_string('name_file_local', 'list_experiments.txt', 'dir to local results')
    flags.DEFINE_string('name_file_remote', 'list_remote_experiments.txt', 'dir to local results')
    flags.DEFINE_string('folder_results', None, 'folder with experiment folder')
    flags.DEFINE_string('transfer_results', None, 'folder with transfer folder')
    flags.DEFINE_string('base_path', None, 'base directory')
    flags.DEFINE_string('output_file_name', 'list_experiments.txt', 'name of the output file')
    flags.DEFINE_string('output_csv_file', 'experiments_information.csv', 'name of the output file')
    flags.DEFINE_string('gt_results_file', None, 'path to the gt values')

    try:
        app.run(main)
    except SystemExit:
        pass
