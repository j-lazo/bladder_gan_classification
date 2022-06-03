import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import oyaml as yaml
import collections


def extract_experiment_information_from_name(experiment_ID_name):

    name_model = experiment_ID_name.split('_fit')[0]
    ending = experiment_ID_name.split('lr_')[1]
    learning_rate = float(ending.split('_bs')[0])
    pre_bs = ending.split('bs_')[1]
    batch_size = int(pre_bs.split('_')[0])
    date = ending.split(''.join(['bs_', str(batch_size), '_']))[1]

    if 'trained_with' in date:
        data_used_in_train = date.split('trained_with_')[-1][:3]
        date = date.split(data_used_in_train + '_')[-1]
    else:
        data_used_in_train = 'ALL'

    if '+' in name_model:
        backbone_name = name_model.split('+')[1]
    else:
        backbone_name = ''

    information_experiment = {'experiment_folder': experiment_ID_name, 'date': date, 'name_model': name_model,
                              'backbones': backbone_name, 'batch_size': batch_size, 'learning_rate': learning_rate,
                              'training_data_used': data_used_in_train}

    return information_experiment


def plot_training_history(list_csv_files, save_dir=''):
    """
    Plots the training history of a model given the list of csv files (in case that there are different training stages)


    Parameters
    ----------
    list_csv_files (list): list of the csv files with the training history
    save_dir (str): The directory where to save the file, if empty, the current working directory

    Returns
    -------

    """
    if len(list_csv_files) > 1:
        print(list_csv_files[0])
        fine_tune_history = pd.read_csv(list_csv_files[0])
        fine_tune_lim = fine_tune_history['epoch'].tolist()[-1]
        header_1 = fine_tune_history.columns.values.tolist()
        train_history = pd.read_csv(list_csv_files[-1])
        header_2 = train_history.columns.values.tolist()
        # mix the headers in case they are different among files
        dictionary = {header_2[i]: name for i, name in enumerate(header_1)}
        train_history.rename(columns=dictionary, inplace=True)
        # append the dataframes in a single one
        train_history = fine_tune_history.append(train_history, ignore_index=True)

    else:
        fine_tune_lim = 0
        train_history = pd.read_csv(list_csv_files[0])

    fig = plt.figure(1, figsize=(12, 9))

    ax1 = fig.add_subplot(221)
    ax1.title.set_text('$ACC$')
    ax1.plot(train_history['accuracy'].tolist(), label='train')
    ax1.plot(train_history['val_accuracy'].tolist(), label='val')
    ax1.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)

    plt.legend(loc='best')

    ax2 = fig.add_subplot(222)
    ax2.title.set_text('PREC')
    ax2.plot(train_history['precision'].tolist(), label='train')
    ax2.plot(train_history['val_precision'].tolist(), label='val')
    ax2.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    ax3 = fig.add_subplot(223)
    ax3.title.set_text('$LOSS$')
    ax3.plot(train_history['loss'].tolist(), label='train')
    ax3.plot(train_history['val_loss'].tolist(), label='val')
    max_xval = np.amax([train_history['loss'].tolist(), train_history['val_loss'].tolist()])
    ax3.fill_between((0, fine_tune_lim), 0, max_xval, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    ax4 = fig.add_subplot(224)
    ax4.title.set_text('$REC$')
    ax4.plot(train_history['recall'].tolist(), label='train')
    ax4.plot(train_history['val_recall'].tolist(), label='val')
    ax4.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
    plt.legend(loc='best')

    if save_dir == '':
        dir_save_figure = os.getcwd() + '/training_history.png'
    else:
        dir_save_figure = save_dir

    print(f'figure saved at: {dir_save_figure}')
    plt.savefig(dir_save_figure)
    plt.close()


def check_path_exists(path, default_ext):
    name, ext = os.path.splitext(path)
    if ext == '':
        if default_ext[0] == '.':
            default_ext = default_ext[1:]
        path = name + '.' + default_ext
    return path


def save_yaml(path, data, **kwargs):

    path = check_path_exists(path, 'yml')

    with open(path, 'w') as f:
        yaml.dump(data, f, **kwargs)


def compute_confusion_matrix(gt_data, predicted_data, plot_figure=False, dir_save_fig=''):
    """
    Compute the confusion Matrix given the ground-truth data (gt_data) and predicted data (predicted_data)
    in list format. If Plot is True shows the matrix .
    Parameters
    ----------
    gt_data : list
    predicted_data : list
    plot_figure :
    dir_save_fig :

    Returns
    -------

    """
    uniques_predicts = np.unique(predicted_data)
    uniques_gt = np.unique(gt_data)
    if collections.Counter(uniques_gt) == collections.Counter(uniques_predicts):
        uniques = uniques_gt
    else:
        uniques = np.unique([*uniques_gt, *uniques_predicts])

    ocurrences = [gt_data.count(unique) for unique in uniques]
    conf_matrix = confusion_matrix(gt_data, predicted_data)
    group_percentages = [conf_matrix[i]/ocurrences[i] for i, row in enumerate(conf_matrix)]

    size = len(list(uniques))
    list_uniques = list(uniques)
    xlabel_names = list()
    for name in list_uniques:
        # if the name of the unique names is longer than 4 characters will split it
        if len(name) > 4:
            name_split = name.split('-')
            new_name = ''
            for splits in name_split:
                new_name = new_name.join([splits[0]])

            xlabel_names.append(new_name)
        else:
            xlabel_names.append(name)

    labels = np.asarray(group_percentages).reshape(size, size)
    sns.heatmap(group_percentages, cmap='Blues', cbar=False, linewidths=.5,
                yticklabels=list(uniques), xticklabels=list(xlabel_names), annot=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Real Class')

    if plot_figure is True:
        plt.show()

    if dir_save_fig == '':
            dir_save_figure = os.getcwd() + '/confusion_matrix.png'

    else:
        if not dir_save_fig.endswith('.png'):
          dir_save_figure = dir_save_fig + 'confusion_matrix.png'
        else:
            dir_save_figure = dir_save_fig

    print(f'figure saved at: {dir_save_figure}')

    plt.savefig(dir_save_figure)
    plt.close()

    return conf_matrix


def analyze_multiclass_experiment(gt_data_file, predictions_data_dir, plot_figure=False, dir_save_figs=None,
                                  analyze_training_history=False, k_folds=None):
    """
    Analyze the results of a multi-class classification experiment

    Parameters
    ----------
    gt_data_file :
    predictions_data_dir :
    plot_figure :
    dir_save_fig :

    Returns
    -------
    History plot, Confusion Matrix

    """
    wli_imgs = []
    nbi_imgs = []
    predictions_nbi = []
    predictions_wli = []
    wli_tissue_types = []
    nbi_tissue_types = []

    list_prediction_files = [f for f in os.listdir(predictions_data_dir) if 'predictions' in f and '(_pre' not in f]
    if k_folds:
        list_prediction_files = [f for f in os.listdir(predictions_data_dir) if 'predictions' in f and '(_pre' not in f
                                 and k_folds in f]

    file_predictiosn = list_prediction_files.pop()
    path_file_predictions = os.path.join(predictions_data_dir, file_predictiosn)
    print(f'file predictions found: {file_predictiosn}')

    df_ground_truth = pd.read_csv(gt_data_file)
    df_preditc_data = pd.read_csv(path_file_predictions)

    predictions_names = df_preditc_data['fname'].tolist()
    predictions_vals = df_preditc_data['over all'].tolist()

    gt_names = df_ground_truth['image_name'].tolist()
    gt_vals = df_ground_truth['tissue type'].tolist()
    imaging_type = df_ground_truth['imaging type'].tolist()

    existing_gt_vals = list()
    ordered_predictiosn = list()
    for name in predictions_names:
        if name in gt_names:
            index = predictions_names.index(name)
            ordered_predictiosn.append(predictions_vals[index])
            index_gt = gt_names.index(name)
            existing_gt_vals.append(gt_vals[index_gt])

            if imaging_type[index_gt] == 'NBI':
                nbi_imgs.append(name)
                predictions_nbi.append(predictions_vals[index])
                nbi_tissue_types.append(gt_vals[index_gt])
            if imaging_type[index_gt] == 'WLI':
                wli_imgs.append(name)
                predictions_wli.append(predictions_vals[index])
                wli_tissue_types.append(gt_vals[index_gt])

    # dir to save the figures
    if dir_save_figs:
        dir_save_fig = dir_save_figs
    else:
        dir_save_fig = predictions_data_dir

    acc_all = accuracy_score(existing_gt_vals, ordered_predictiosn)
    acc_wli = accuracy_score(wli_tissue_types, predictions_wli)
    acc_nbi = accuracy_score(nbi_tissue_types, predictions_nbi)

    macro_prec_all = precision_score(existing_gt_vals, ordered_predictiosn, average='macro', zero_division=1)
    macro_prec_wli = precision_score(wli_tissue_types, predictions_wli, average='macro', zero_division=1)
    marco_prec_nbi = precision_score(nbi_tissue_types, predictions_nbi, average='macro', zero_division=1)

    macro_rec_all = recall_score(existing_gt_vals, ordered_predictiosn, average='macro', zero_division=1)
    macro_rec_wli = recall_score(wli_tissue_types, predictions_wli, average='macro', zero_division=1)
    macro_rec_nbi = recall_score(nbi_tissue_types, predictions_nbi, average='macro', zero_division=1)

    macro_f1_all = f1_score(existing_gt_vals, ordered_predictiosn, average='macro', zero_division=1)
    macro_f1_wli = f1_score(wli_tissue_types, predictions_wli, average='macro', zero_division=1)
    macro_f1_nbi = f1_score(nbi_tissue_types, predictions_nbi, average='macro', zero_division=1)

    performance_resume = {'Accuracy ALL': float(acc_all),
                          'Accuracy WLI': float(acc_wli),
                          'Accuracy NBI': float(acc_nbi),
                          'Precision ALL': float(macro_prec_all),
                          'Precision WLI': float(macro_prec_wli),
                          'Precision NBI': float(marco_prec_nbi),
                          'Recall ALL': float(macro_rec_all),
                          'Recall WLI': float(macro_rec_wli),
                          'Recall NBI': float(macro_rec_nbi),
                          'F-1 ALL': float(macro_f1_all),
                          'F-1 WLI': float(macro_f1_wli),
                          'F-1 NBI': float(macro_f1_nbi),
                          }

    if k_folds:
        yaml_file_name = ''.join(['performance_analysis_', k_folds, '.yaml'])
        name_fig_1 = ''.join(['confusion_matrix_all_', k_folds, '.png'])
        name_fig_2 = ''.join(['confusion_matrix_wli_', k_folds, '.png'])
        name_fig_3 = ''.join(['confusion_matrix_nbi_', k_folds, '.png'])
        save_hist_fig_name = ''.join(['training_history_', k_folds, '.png'])
    else:
        yaml_file_name = 'performance_analysis.yaml'
        name_fig_1 = 'confusion_matrix_all.png'
        name_fig_2 = 'confusion_matrix_wli.png'
        name_fig_3 = 'confusion_matrix_nbi.png'
        save_hist_fig_name = 'training_history.png'

    dir_data_yaml = os.path.join(dir_save_fig, yaml_file_name)
    dir_save_hist = os.path.join(dir_save_fig, save_hist_fig_name)
    save_yaml(dir_data_yaml, performance_resume)

    # Confusion Matrices

    compute_confusion_matrix(existing_gt_vals, ordered_predictiosn, plot_figure=False,
                             dir_save_fig=dir_save_fig + name_fig_1)

    compute_confusion_matrix(wli_tissue_types, predictions_wli,
                             dir_save_fig=dir_save_fig + name_fig_2)

    compute_confusion_matrix(nbi_tissue_types, predictions_nbi,
                             dir_save_fig=dir_save_fig + name_fig_3)

    gt_values = []
    for name in predictions_names:
        if name in gt_names:
            index = gt_names.index(name)
            gt_values.append(gt_vals[index])

    new_df = df_preditc_data.copy()
    data_top = list(new_df.columns)
    new_df.insert(len(data_top), "real values", gt_values, allow_duplicates=True)
    name_data_save = path_file_predictions
    new_df.to_csv(name_data_save, index=False)
    print(f'results saved at {name_data_save}')

    # analyze the history
    if analyze_training_history is True:
        list_history_files = [f for f in os.listdir(predictions_data_dir) if 'train_history' in f]
        ordered_history = list()
        fine_tune_file = [f for f in list_history_files if 'fine_tune' in f]
        if fine_tune_file:
            fine_tune_file_dir = predictions_data_dir + fine_tune_file.pop()
            ordered_history.append(fine_tune_file_dir)

        ordered_history.append(predictions_data_dir + list_history_files[-1])
        plot_training_history(ordered_history, save_dir=dir_save_hist)

    return performance_resume


def boxplot_individual_cases(data_frame, columns_header, title_plot=''):
    print(data_frame)
    fig1 = plt.figure(1, figsize=(11, 7))
    fig1.canvas.set_window_title(title_plot)
    fig1.suptitle(title_plot, fontsize=14)
    ax1 = fig1.add_subplot(131)
    ax1 = sns.boxplot(x=columns_header[0], y=columns_header[1], data=data_frame)
    ax1 = sns.swarmplot(x=columns_header[0], y=columns_header[1], data=data_frame, color=".25")
    ax1.title.set_text('ALL')

    ax2 = fig1.add_subplot(132)
    ax2 = sns.boxplot(x=columns_header[0], y=columns_header[2], data=data_frame)
    ax2 = sns.swarmplot(x=columns_header[0], y=columns_header[2], data=data_frame, color=".25")
    ax2.title.set_text('WLI')

    ax3 = fig1.add_subplot(133)
    ax3 = sns.boxplot(x=columns_header[0], y=columns_header[3], data=data_frame)
    ax3 = sns.swarmplot(x=columns_header[0], y=columns_header[3], data=data_frame, color=".25")
    ax3.title.set_text('NBI')

    plt.show()


def perform_global_analysis(path_results=os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2'),
                            output_file=os.path.join(os.getcwd(), 'results', 'results_resume.csv'),
                            gt_file=''):

    # define all the lists:
    name_experiment = list()
    date_experiment = list()
    name_model = list()
    backbones = list()
    learning_rate = ()
    acc_all = list()
    acc_nbi = list()
    acc_wli = list()

    if os.path.isfile(output_file):
        df = pd.read_csv(output_file)
        previous_experiments = df['folder name'].tolist()
        date = df['date'].tolist()

    else:
        df = None
        previous_experiments = list()

    list_experiments = [f for f in os.listdir(path_results)
                        if os.path.isdir(os.path.join(path_results, f))]

    for i, experiment in enumerate(tqdm.tqdm(list_experiments, desc='Analysing folders')):
        if experiment not in previous_experiments:
            pass
            # read predictions file
            # analyze data"""

    # merge previous and new results
    # sort by name
    # save results
    #DataFrame.to_csv(output_file)
