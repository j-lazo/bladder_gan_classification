import os
import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
import shutil
import tqdm
import numpy as np
import copy
from utils.data_management import file_management as dam


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
    dir_to_analyze = FLAGS.dir_to_analyze

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
                dam.compress_files(path_folder, destination_dir=compressed_results)
                # make a temporal folder with only the files you want to compress
                temporal_folder_dir = os.path.join(os.getcwd(), 'results', 'remove_folders', experiment_folder)
                os.mkdir(temporal_folder_dir)
                # move files to temporal folder
                list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                              f.endswith('.yaml') or f.endswith('.csv')]
                # temporal folder to delete
                for file_name in list_files:
                    shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
                # compress the file
                dam.compress_files(temporal_folder_dir, destination_dir=transfer_results)

    elif mode == 'update_experiment_list':

        if base_path:
            directory_path = os.path.join(base_path, 'results', 'bladder_tissue_classification_v2')
        else:
            directory_path = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')

        list_files = [f for f in os.listdir(directory_path)
                      if os.path.isdir(os.path.join(directory_path, f)) or f.endswith('.zip')]
        # remove repeated zip a and/or dir files
        unique_files = np.unique([f.replace('.zip', '') for f in list_files])
        list_existing_files = dam.check_file_exists(directory_path, name_file=output_file_name)
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

    elif mode == 'unizp':

        # unzip the files that haven't been unzip
        dir_zipped_files = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_transfer')
        dir_unzipped_files = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')

        list_transfered_files = [f for f in os.listdir(dir_zipped_files) if f.endswith('.zip')]
        list_unzipped_files = [f for f in os.listdir(dir_unzipped_files) if os.path.isdir(os.path.join(dir_unzipped_files, f))]

        for i, zipped_file in enumerate(tqdm.tqdm(list_transfered_files, desc='Preparing files')):
            if zipped_file.replace('.zip', '') not in list_unzipped_files:
                filename = os.path.join(dir_zipped_files, zipped_file)
                dir_unzipped = os.path.join(dir_unzipped_files, zipped_file.replace('.zip', ''))
                if not os.path.isdir(dir_unzipped):
                    os.mkdir(dir_unzipped)
                print(f'Expanding {zipped_file}  ...')
                shutil.unpack_archive(filename, dir_unzipped)

    elif mode == 'analyze':

        # analyze the yamls files
        if not dir_to_analyze:
            dir_unzipped_files = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2')
        name_experiment = list()
        date_experiment = list()
        name_models = list()
        backbones = list()

        dictionary_experiments = {}

        list_experiments = [f for f in os.listdir(dir_unzipped_files) if os.path.isdir(os.path.join(dir_unzipped_files, f))]
        for j, experiment_folder in enumerate(tqdm.tqdm(list_experiments, desc='Preparing files')):
            # extract the information from the folder's

            yaml_files = [f for f in os.listdir(os.path.join(dir_unzipped_files, experiment_folder)) if f.endswith('.yaml')]

            if yaml_files:
                if 'experiment_information.yaml' in yaml_files:
                    performance_yaml_file = 'performance_analysis.yaml'
                    experiment_information_yaml = 'experiment_information.yaml'
                    path_yaml_information = os.path.join(dir_unzipped_files, experiment_folder,
                                                         experiment_information_yaml)
                    info_experiment = dam.read_yaml_file(path_yaml_information)
                    date = info_experiment['date']
                    name_model = info_experiment['name model']
                    backbone_name = info_experiment['backbone']
                    batch_size = info_experiment['batch size']
                    learning_rate = info_experiment['learning rate']
                    data_used_in_train = info_experiment['domain data used']
                    backbone_gan = info_experiment['backbone gan']
                    dataset_training = info_experiment['dataset']
                    name_experiment.append(experiment_folder)
                    date_experiment.append(date)
                    name_models.append(name_model)
                    backbones.append(backbone_name)

                else:
                    performance_yaml_file = yaml_files.pop()
                    # extract the information from the folder's name

                    name_model = experiment_folder.split('_fit')[0]
                    ending = experiment_folder.split('lr_')[1]
                    learning_rate = float(ending.split('_bs')[0])
                    pre_bs = ending.split('bs_')[1]
                    batch_size = int(pre_bs.split('_')[0])
                    date = ending.split(''.join(['bs_', str(batch_size), '_']))[1]
                    if 'trained_with' in date:
                        data_used_in_train = date.split('trained_with_')[-1][:3]
                        date = date.split(data_used_in_train + '_')[-1]
                    else:
                        data_used_in_train = 'ALL'

                    dataset_training = 'bladder_tissue_classification_v2'
                    backbone_gan = 'checkpoint_charlie'
                    name_experiment.append(experiment_folder)
                    date_experiment.append(date)
                    name_models.append(name_model)

                    if '+' in name_model:
                        backbone_name = name_model.split('+')[1]
                        backbones.append(backbone_name)
                    else:
                        backbone_name = ''
                        backbones.append('')

                path_yaml_performance = os.path.join(dir_unzipped_files, experiment_folder, performance_yaml_file)

                results = dam.read_yaml_file(path_yaml_performance)

                acc_all = float(results['Accuracy ALL'])
                acc_nbi = float(results['Accuracy NBI'])
                acc_wli = float(results['Accuracy WLI'])

                prec_all = float(results['Precision ALL'])
                prec_nbi = float(results['Precision NBI'])
                prec_wli = float(results['Precision WLI'])

                rec_all = float(results['Recall ALL'])
                rec_nbi = float(results['Recall NBI'])
                rec_wli = float(results['Recall WLI'])

                f1_all = float(results['F-1 ALL'])
                f1_nbi = float(results['F-1 NBI'])
                f1_wli = float(results['F-1 WLI'])

            else:
                acc_all = None
                acc_nbi = None
                acc_wli = None
                prec_all = None
                prec_nbi = None
                prec_wli = None
                rec_all = None
                rec_nbi = None
                rec_wli = None
                f1_all = None
                f1_nbi = None
                f1_wli = None

            if data_used_in_train == '':
                data_used_in_train = 'ALL'
            information_experiment = {'experiment_folder': experiment_folder, 'date': date, 'name_model': name_model,
                                      'backbones': backbone_name, 'batch_size': batch_size,
                                      'learning_rate': learning_rate, 'dataset': dataset_training,
                                      'backbone GAN': backbone_gan, 'training_data_used': data_used_in_train,
                                      'Accuracy ALL': acc_all, 'Accuracy WLI': acc_wli, 'Accuracy NBI': acc_nbi,
                                      'Precision ALL': prec_all, 'Precision WLI': prec_nbi, 'Precision NBI': prec_wli,
                                      'Recall ALL': rec_all, 'Recall WLI': rec_nbi, 'Recall NBI': rec_wli,
                                      'F-1 ALL': f1_all, 'F-1 WLI': f1_nbi, 'F-1 NBI': f1_wli,
                                      }

            dictionary_experiments.update({experiment_folder: information_experiment})

        print(np.unique(name_models))
        list_acc_all = [dictionary_experiments[experiment]['Accuracy ALL'] for experiment in dictionary_experiments]
        list_acc_nbi = [dictionary_experiments[experiment]['Accuracy NBI'] for experiment in dictionary_experiments]
        list_acc_wli = [dictionary_experiments[experiment]['Accuracy WLI'] for experiment in dictionary_experiments]

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

        #daa.analyze_individual_cases(output_csv_file_dir)

    elif mode == 'transfer_results':
        list_files = os.listdir(local_path_results)

        # first compare which files are missing in local
        list_missing = list()
        if name_file_local in list_files:
            list_files_local = dam.read_txt_file(os.path.join(local_path_results, name_file_local))
            list_files_remote = dam.read_txt_file(os.path.join(local_path_results, name_file_remote))
            list_missing = [f for f in list_files_remote if f not in list_files_local]
            print(f'{len(list_missing)} need to be sync')

        for i, missing_file in enumerate(tqdm.tqdm(list_missing, desc='Preparing files')):
            print(f'Coping {missing_file} to local')
            dam.transfer_files(local)


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
    flags.DEFINE_string('dir_to_analyze', None, 'path to analyze all the experiements')

    try:
        app.run(main)
    except SystemExit:
        pass
