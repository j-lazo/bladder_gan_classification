import shutil

from absl import app, flags
from absl.flags import FLAGS
import os
import tqdm
import numpy as np
import pandas as pd
from utils import data_analysis as daa
from utils.data_management import file_management as fam


def compute_metrics_batch_experiments(dir_experiment_files, dir_output_csv_file_dir = os.getcwd(),
                                      gt_file=os.path.join(os.getcwd(), 'annotations_test.csv')):

    """
    Generates a CSV file resuming the main metrics of each experiment
    :param dir_experiment_files: absolute path to the directory where all the experiment folders are saved, it expects
    that  inside each folder there is a .yaml file with the resume of the metrics, if not....
    :type dir_experiment_files: str
    :return:
    :rtype:
    """

    output_csv_file = os.path.join(dir_output_csv_file_dir, 'results', 'experiment_performances.csv')
    name_experiment = list()
    dictionary_experiments = {}
    list_experiments = [f for f in os.listdir(dir_experiment_files) if os.path.isdir(os.path.join(dir_experiment_files, f))]
    for j, experiment_folder in enumerate(tqdm.tqdm(list_experiments[:], desc='Preparing files')):

        # extract the information from the folder's name
        information_experiment = daa.extract_experiment_information_from_name(experiment_folder)
        name_experiment.append(experiment_folder)
        predictions_data_dir = os.path.join(dir_experiment_files, experiment_folder)
        performance_resume = daa.analyze_multiclass_experiment(gt_file, predictions_data_dir, plot_figure=False)
        resume_experiment = {**information_experiment, **performance_resume}
        dictionary_experiments.update({experiment_folder: resume_experiment})

        dir_data_yaml = os.path.join(predictions_data_dir, 'performance_analysis.yaml')
        fam.save_yaml(dir_data_yaml, performance_resume)

    new_df = pd.DataFrame.from_dict(dictionary_experiments, orient='index')
    output_csv_file_dir = os.path.join(os.getcwd(), 'results', output_csv_file)
    new_df.to_csv(output_csv_file_dir, index=False)
    print(f'results saved at {output_csv_file_dir}')

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

    sorted_df = pd.DataFrame.from_dict(sorted_dict, orient='index')
    output_csv_file_dir2 = os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')
    sorted_df.to_csv(output_csv_file_dir2, index=False)

    sorted_df_nbi = pd.DataFrame.from_dict(srted_nbi, orient='index')
    output_csv_file_dir3 = os.path.join(os.getcwd(), 'results', 'sorted_nbi_experiments_information.csv')
    sorted_df_nbi.to_csv(output_csv_file_dir3, index=False)

    sorted_df_wli = pd.DataFrame.from_dict(sorted_wli, orient='index')
    output_csv_file_dir4 = os.path.join(os.getcwd(), 'results', 'sorted_wli_experiments_information.csv')
    sorted_df_wli.to_csv(output_csv_file_dir4, index=False)

    return output_csv_file_dir


def compute_metrics_boxplots(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):

    metrics = ['Accuracy', 'Precision', 'Recall', 'F-1']
    df = pd.read_csv(dir_to_csv)
    name_models = df['name_model'].tolist()
    batch_sizes = df['batch_size'].tolist()
    learning_rates = df['learning_rate'].tolist()
    data_used = df['training_data_used'].tolist()
    training_data_used = df['training_data_used'].tolist()

    unique_models = np.unique(name_models)
    print(f'unique models: {unique_models}')
    alternative_name = {'gan_model_joint_features_and_domain+densenet121_': 'jf+d D',
                        'gan_model_multi_joint_features+densenet121_': 'jf D',
                        'gan_model_multi_joint_features': 'jf+d-R',
                        'gan_model_multi_joint_features+resnet101_': 'jf R',
                        'densenet121': 'densenet',
                        'gan_model_separate_features+densenet121_': 'sf D',
                        'gan_model_separate_features': 'sf R',
                        'resnet101': 'resnet101',
                        'simple_model_domain_input+resnet101_': 'smd R',
                        'simple_model_domain_input+densenet121_': 'smd D',
                        'simple_separation_model+resnet101_': 'ss R',
                        'gan_model_separate_features_v2+resnet101_': 'sf R v2', 'gan_model_separate_features_v2': 'sf2',
                        'gan_model_joint_features_and_domain': 'g-mjf-d',
                        'gan_model_separate_features_v3': 'sf v3',
                        'simple_model_domain_input': 'smdi',
                        'simple_separation_model': 'ssm',
                        'simple_model_with_backbones': 'smwb'}

    selected_models_basic = ['densenet121', 'resnet101']
    selected_models_joint = ['gan_model_separate_features+densenet121_',
                       'gan_model_separate_features+resnet101_',
                       'gan_model_multi_joint_features+resnet101_',
                       'gan_model_multi_joint_features+densenet121_']

    selected_models_d = ['densenet121', 'resnet101',
                       'gan_model_separate_features',
                        'gan_model_multi_joint_features',]
                       #'gan_model_joint_features_and_domain+densenet121_',]
                         #'simple_model_domain_input+densenet121_']

    selected_models_r = ['resnet101', 'densenet121',
                       'gan_model_separate_features',
                       'gan_model_multi_joint_features',
                         'simple_separation_model+resnet101_',
                         'gan_model_separate_features_v2+resnet101_']
                       #'gan_model_joint_features_and_domain+resnet101_',]
                         #'simple_model_domain_input+resnet101_']

    selected_models_c = ['gan_model_separate_features+densenet121_',
                         'gan_model_multi_joint_features+densenet121_',
                         'gan_model_joint_features_and_domain+densenet121_'
                         'gan_model_separate_features+resnet101_',
                         'gan_model_multi_joint_features+resnet101_',
                         'gan_model_joint_features_and_domain+resnet101_'
                         ]
    selected_res = ['gan_model_separate_features', 'resnet101', 'simple_separation_model', 'simple_model_with_backbones',
                    'simple_model_domain_input']
    selected_models_a = ['resnet101', 'gan_model_separate_features', 'gan_model_multi_joint_features',
                          'gan_model_joint_features_and_domain',
                          'simple_model_domain_input',
                         'simple_model_with_backbones', 'simple_separation_model']
    select_b = ['resnet101']
    #selected_models_a = ['gan_model_multi_joint_features']
    selected_models = selected_models_a

    list_x = list()
    list_y1 = list()
    list_y2 = list()
    list_y3 = list()
    list_domain = list()
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    chosen_dataset = ['bladder_tissue_classification_v3']
                      #'bladder_tissue_classification_v3_augmented',
                      #'bladder_tissue_classification_v3_gan_new',]

    chosen_trained_data = ['ALL', 'WLI']

    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,]
                            #'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi']
    dictionary_selection = {'name_model': selected_models, 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data}
    selection = select_specific_cases(df, dictionary_selection)

    # 'Accuracy', 'Precision', 'Recall', 'F-1'
    metric_analysis = 'F-1'
    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    lis_metric_1 = df[''.join([metric_analysis, ' ', 'ALL'])].tolist()
    lis_metric_2 = df[''.join([metric_analysis, ' ', 'WLI'])].tolist()
    lis_metric_3 = df[''.join([metric_analysis, ' ', 'NBI'])].tolist()

    for j, model in enumerate(name_models):
        if model in selected_models \
                and learning_rates[j] in chosen_learning_rates \
                and batch_sizes[j] in chosen_batch_sizes \
                and training_data_used[j] in chosen_trained_data:

            model_name = alternative_name[model]
            list_x.append(model_name)
            list_y1.append(lis_metric_1[j])
            list_y2.append(lis_metric_2[j])
            list_y3.append(lis_metric_3[j])
            list_domain.append(data_used[j])

    zipped = list(zip(list_x, list_y1, list_y2, list_y3, list_domain))
    columns = ['name_model'] + metrics_box
    x_axis = 'name_model'
    y_axis = metrics_box
    columns.append('training_data_used')
    print(columns)
    df_analysis = pd.DataFrame(zipped, columns=columns)
    title_plot = 'Comparison base models'#name_model.replace('_', ' ')
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')
    title_fig = ''.join([metric_analysis, ' ', 'trained using ', chosen_trained_data[0]])
    #daa.box_plot_matplotlib(df_analysis, metrics=metrics_box, title=title_fig, y_label=metric_analysis,
    #                        analysis_type='by_model')


def select_specific_cases(data_frame, dictionary_selection):
    """
    Based on the keys of a given dictionary, returns a dataframe which fulfills the conditions given by the
    dictionary
    :param data_frame:
    :type data_frame:
    :param dictionary_selection:
    :type dictionary_selection:
    :return:
    :rtype:
    """

    print(data_frame)
    keys_df = data_frame.keys().tolist()
    new_dict = {k: dictionary_selection[k] for k in dictionary_selection if k in keys_df}
    result_df = data_frame.copy()
    print(result_df)
    for k in new_dict:
        result_df = result_df[result_df[k].isin(new_dict[k])]

    return result_df


def prepare_data(dir_dataset, destination_directory=os.path.join(os.getcwd(), 'datasets', 'bladder_tissue_classification_k_folds')):
    """

    :param dir_dataset: dataset with case-wise divided folders
    :type dir_dataset: str
    :param destination_directory: directory where to save the data
    :type destination_directory: str
    :return:
    :rtype:
    """

    list_sub_folders = [f for f in os.listdir(dir_dataset) if os.path.isdir(os.path.join(dir_dataset, f))]

    headers = ['image_name', 'case number', 'useful', 'clear', 'resolution', 'anatomical region', 'imaging type',
               'type artifact', 'tissue type', 'fov shape', 'ROI']
    new_df = pd.DataFrame(columns=headers)
    list_df = [new_df]
    for folder_name in list_sub_folders:

        path_folder = os.path.join(dir_dataset, folder_name)
        csv_files = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
        df = pd.read_csv(os.path.join(path_folder, csv_files.pop()))
        df.insert(1, "case number", None)
        list_keys = df.keys().tolist()
        list_keys.insert(1, 'case number')
        list_images = df['image_name'].tolist()
        list_tissues = df['tissue type'].tolist()
        for image_name in list_images:
            index = list_images.index(image_name)
            row = df.loc[index]
            if row['image_name'] == image_name:
                row['case number'] = folder_name
                df.loc[index] = row

        list_df.append(df)
        # copy images
        list_sub_dirs_cases = [os.path.join(path_folder, f) for f in os.listdir(path_folder)
                               if os.path.isdir(os.path.join(path_folder, f))]
        for sub_dirs_case in list_sub_dirs_cases:
            list_imgs = [f for f in os.listdir(sub_dirs_case) if f.endswith('.png')]
            for j, image in enumerate(tqdm.tqdm(list_imgs[:], desc='Moving images')):
                if image in list_images:
                    index_image = list_images.index(image)
                    origin_dir = os.path.join(sub_dirs_case, image)
                    destination_dir = os.path.join(destination_directory, list_tissues[index_image], image)
                    shutil.copyfile(origin_dir, destination_dir)

    # merge df
    resulting_df = pd.concat(list_df)
    final_list_tissues = resulting_df['tissue type'].tolist()
    cases_final = {f: final_list_tissues.count(f) for f in np.unique(final_list_tissues)}
    print(cases_final)
    save_path_csv = os.path.join(destination_directory, 'annotations.csv')
    print(f'New CSV file saved at {save_path_csv}')
    resulting_df.to_csv(save_path_csv, index=False)


def analyze_composition_dataset(dir_dataset):

    list_sub_folders = [f for f in os.listdir(dir_dataset) if os.path.isdir(os.path.join(dir_dataset, f))]
    folds_dict = {'fold_1': [26, 14, 25, 12, 18, 17, 16, 32],
                  'fold_2': [3, 28, 23, 20, 9, 8, 31],
                  'fold_3': [27, 21, 6, 13, 11, 4, 2, 30],
                  'fold_4': [1, 24, 22, 7, 5, 10, 29]}

    for fold in folds_dict:
        num_imgs = 0
        list_cases = folds_dict[fold]
        num_imgs_csv = 0
        fold_cases = {'CIS': 0, 'HGC': 0, 'LGC': 0, 'HLT': 0, 'NTL': 0}
        domain_cases = {'WLI': 0, 'NBI': 0}
        for num_case in list_cases:
            folder_name = ''.join(['case_', str(num_case).zfill(3)])
            list_sub_folders.remove(folder_name)
            path_folder = os.path.join(dir_dataset, folder_name)
            csv_files = [f for f in os.listdir(path_folder) if f.endswith('.csv')]
            df = pd.read_csv(os.path.join(path_folder, csv_files.pop()))
            list_images = df['image_name'].tolist()
            list_tissues = df['tissue type'].tolist()
            list_domains = df['imaging type'].tolist()
            num_imgs_csv = num_imgs_csv + len(list_images)

            cases_dir = {f: list_tissues.count(f) for f in np.unique(list_tissues)}
            domain_dir = {f: list_domains.count(f) for f in np.unique(list_domains)}
            for cases in cases_dir:
                fold_cases[cases] = fold_cases[cases] + cases_dir[cases]
            for domains in domain_dir:
                domain_cases[domains] = domain_cases[domains] + domain_dir[domains]

            list_sub_dirs_cases = [os.path.join(path_folder, f) for f in os.listdir(path_folder)
                                   if os.path.isdir(os.path.join(path_folder, f))]
            for sub_dirs_case in list_sub_dirs_cases:
                list_imgs = [f for f in os.listdir(sub_dirs_case) if f.endswith('.png')]
                num_imgs_sub_dir = len(list_imgs)
                num_imgs = num_imgs + num_imgs_sub_dir

        print(f'a total of {len(list_cases)} cases were found in {fold} and in total {num_imgs} or {num_imgs_csv} iamges were found')
        print(fold_cases)
        print(domain_cases)


def main(_argv):

    type_of_analysis = FLAGS.type_of_analysis
    directory_to_analyze = FLAGS.directory_to_analyze
    path_file_to_analyze = FLAGS.path_file_to_analyze
    ground_truth_file = FLAGS.ground_truth_file
    plot_figures = FLAGS.plot_figures
    display_figures = FLAGS.display_figures

    if type_of_analysis:
        if type_of_analysis == 'compute_metrics_batch_experiments':
            compute_metrics_batch_experiments(directory_to_analyze, gt_file=ground_truth_file)
        elif type_of_analysis == 'compute_metrics_boxplots':
            compute_metrics_boxplots()
        elif type_of_analysis == 'analyze_composition_dataset':
            analyze_composition_dataset(directory_to_analyze)
        elif type_of_analysis == 'prepare_data':
            prepare_data(directory_to_analyze)


if __name__ == '__main__':

    flags.DEFINE_string('type_of_analysis', None, 'Select option of analysis to perform')
    flags.DEFINE_string('directory_to_analyze', None, 'Absolute path to the directory with the files needed to analyze')
    flags.DEFINE_string('file_to_analyze', None, 'In case only a single file is going to be analyzed')
    flags.DEFINE_string('ground_truth_file', os.path.join(os.getcwd(), 'annotations_test.csv'),
                        'Absolute path to the file with the ground-truth file')
    flags.DEFINE_boolean('plot_figures', False, 'in case the function plots figures')
    flags.DEFINE_boolean('display_figures', False, 'in case the function plots figures and you want to visualize them')
    flags.DEFINE_string('dir_output_csv_file_dir', os.getcwd(), 'where to save the output file in case the analysis'
                                                                'produces an output file')
    flags.DEFINE_string('path_file_to_analyze', '', 'File to perform some analysis')

    try:
        app.run(main)
    except SystemExit:
        pass
