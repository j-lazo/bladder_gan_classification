import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tqdm
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import oyaml as yaml
import collections
from matplotlib.patches import Polygon
import scipy
from utils.data_management import file_management as fam
from matplotlib import gridspec
from decimal import Decimal

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
        header = train_history.columns.values.tolist()

    fig = plt.figure(1, figsize=(12, 9))
    if 'accuracy' in header:

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

    elif 'final_output_accuracy' in header:
        fig = plt.figure(1, figsize=(12, 9))

        ax1 = fig.add_subplot(221)
        ax1.title.set_text('$ACC$')
        ax1.plot(train_history['final_output_accuracy'].tolist(), label='train')
        ax1.plot(train_history['val_final_output_accuracy'].tolist(), label='val')
        ax1.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)

        plt.legend(loc='best')

        ax2 = fig.add_subplot(222)
        ax2.title.set_text('PREC')
        ax2.plot(train_history['final_output_precision'].tolist(), label='train')
        ax2.plot(train_history['val_final_output_precision'].tolist(), label='val')
        ax2.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
        plt.legend(loc='best')

        ax3 = fig.add_subplot(223)
        ax3.title.set_text('$LOSS$')
        ax3.plot(train_history['final_output_loss'].tolist(), label='train')
        ax3.plot(train_history['val_final_output_loss'].tolist(), label='val')
        max_xval = np.amax([train_history['loss'].tolist(), train_history['val_loss'].tolist()])
        ax3.fill_between((0, fine_tune_lim), 0, max_xval, facecolor='orange', alpha=0.4)
        plt.legend(loc='best')

        ax4 = fig.add_subplot(224)
        ax4.title.set_text('$REC$')
        ax4.plot(train_history['final_output_recall'].tolist(), label='train')
        ax4.plot(train_history['val_final_output_recall'].tolist(), label='val')
        ax4.fill_between((0, fine_tune_lim), 0, 1, facecolor='orange', alpha=0.4)
        plt.legend(loc='best')

    if save_dir == '':
        dir_save_figure = os.getcwd() + '/training_history.png'
    else:
        dir_save_figure = save_dir

    print(f'figure saved at: {dir_save_figure}')
    plt.savefig(dir_save_figure)
    plt.close()


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


def group_data_tumor():
    tumor_classes = ['CIS', 'HGC', 'LGC']
    pass


def group_data_lesion():
    lesion_classes = ['CIS', 'HGC', 'LGC', 'NTL']
    pass


def analyze_multiclass_experiment(gt_data_file, predictions_data_dir, plot_figure=True, dir_save_figs=None,
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
    unique_classes = np.unique(gt_vals)

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
    per_class_precision_all = precision_score(existing_gt_vals, ordered_predictiosn, average=None, zero_division=1)
    precisions_per_class = {f'PREC {unique_classes[j]}': float(metric_result) for j, metric_result in enumerate(per_class_precision_all)}

    macro_rec_all = recall_score(existing_gt_vals, ordered_predictiosn, average='macro', zero_division=1)
    macro_rec_wli = recall_score(wli_tissue_types, predictions_wli, average='macro', zero_division=1)
    macro_rec_nbi = recall_score(nbi_tissue_types, predictions_nbi, average='macro', zero_division=1)
    per_class_recall_all = recall_score(existing_gt_vals, ordered_predictiosn, average=None, zero_division=1)
    recall_per_class = {f'REC {unique_classes[j]}': float(metric_result) for j, metric_result in enumerate(per_class_recall_all)}

    macro_f1_all = f1_score(existing_gt_vals, ordered_predictiosn, average='macro', zero_division=1)
    macro_f1_wli = f1_score(wli_tissue_types, predictions_wli, average='macro', zero_division=1)
    macro_f1_nbi = f1_score(nbi_tissue_types, predictions_nbi, average='macro', zero_division=1)
    per_class_f1_all = f1_score(existing_gt_vals, ordered_predictiosn, average=None, zero_division=1)
    f1_per_class = {f'F-1 {unique_classes[j]}': float(metric_result) for j, metric_result in enumerate(per_class_f1_all)}



    cohen_kappa_score_all = cohen_kappa_score(existing_gt_vals, ordered_predictiosn)
    matthews_corrcoef_all = matthews_corrcoef(existing_gt_vals, ordered_predictiosn)
    matthews_corrcoef_wli = matthews_corrcoef(wli_tissue_types, predictions_wli)
    matthews_corrcoef_nbi = matthews_corrcoef(nbi_tissue_types, predictions_nbi)

    performance_resume = {'Accuracy ALL': float(acc_all),
                          'Accuracy WLI': float(acc_wli),
                          'Accuracy NBI': float(acc_nbi),
                          'Cohen Kappa': float(cohen_kappa_score_all),
                          'Matthews CC ALL': float(matthews_corrcoef_all),
                          'Matthews CC WLI': float(matthews_corrcoef_wli),
                          'Matthews CC NBI': float(matthews_corrcoef_nbi),
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

    performance_resume = {**performance_resume, **precisions_per_class}
    performance_resume = {**performance_resume, **recall_per_class}
    performance_resume = {**performance_resume, **f1_per_class}

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
    fam.save_yaml(dir_data_yaml, performance_resume)

    if plot_figure:
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


def compute_Mann_Whitney_U_test(data_model_1, data_model_2, metric_list=None, comparison_attribute='name_model'):

    name_1 = list(np.unique(data_model_1[comparison_attribute].tolist()))[-1]
    name_2 = list(np.unique(data_model_2[comparison_attribute].tolist()))[-1]

    training_data_used = list(np.unique(data_model_1['training_data_used'].tolist()))
    print(f'{name_1} vs {name_2}')
    for metric in metric_list:
        for training_data in training_data_used:
            print(f'{metric} trained with {training_data}')
            df1 = data_model_1.loc[data_model_1['training_data_used'] == training_data]
            df2 = data_model_2.loc[data_model_2['training_data_used'] == training_data]
            list_metric_model_1 = df1[metric].tolist()
            list_metric_model_2 = df2[metric].tolist()
            result_mwu_test = scipy.stats.mannwhitneyu(list_metric_model_1, list_metric_model_2)
            print(f'{name_1} : {np.median(list_metric_model_1)} +- {np.std(list_metric_model_1)}')
            print(f'{name_2} : {np.median(list_metric_model_2)} +- {np.std(list_metric_model_2)}')
            print(f'stat-significance: {result_mwu_test}')
            print(type(result_mwu_test))
            print(len(result_mwu_test))
            print(np.shape(result_mwu_test))
            print(result_mwu_test[1])


def analyze_diagnosis_experiment(df_real_vals, df_results):

    dict_names = {'Low Grade': 'LGC',
                  'Non-Tumor Lesion': 'NTL',
                  'Non-Suspicious Tissue': 'HLT',
                  'High Grade': 'HGC'}

    list_acc_base = list()
    list_prec_base = list()
    list_rec_base = list()
    list_f1_base = list()
    list_mathews_base = list()
    list_acc_prop = list()
    list_prec_prop = list()
    list_rec_prop = list()
    list_f1_prop = list()
    list_mathews_prop = list()

    real_vals = list()
    diagnosis_type = list()
    keys_of_interest = list(range(21, 60))
    question_num = df_real_vals['Question'].tolist()
    type_class = df_real_vals['REAL CLASS'].tolist()
    data_type = df_real_vals['Type'].tolist()

    for j, num in enumerate(question_num):
        if not np.isnan(num):
            if int(num) in keys_of_interest:
                real_vals.append(type_class[j])
                diagnosis_type.append(data_type[j])

    unique_vals = list(np.unique(real_vals))
    real_vals = [unique_vals.index(x) for x in real_vals]
    real_vals_prop = [x for j, x in enumerate(real_vals) if diagnosis_type[j] == 'mixed']
    real_vals_base = [x for j, x in enumerate(real_vals) if diagnosis_type[j] == 'temporal frame']

    keys_of_interest = ['Q'+str(j)+'.' for j in keys_of_interest]

    for j in range(len(df_results['Timestamp'].tolist())):
        list_results_base = list()
        list_results_proposed = list()
        case_dict = df_results.iloc[j].to_dict()
        new_dict = {k.replace(' ', ''):v for k, v in case_dict.items()}
        new_dict = {k.replace('Pleaseselecttheimagethatlooksmorerealistic', ''):v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythattheimagescorrespondto.', ''): v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythatyouthinktheimagesbelongsto.', ''): v for k, v in new_dict.items()}

        for j, key in enumerate(keys_of_interest):
            if diagnosis_type[j] == 'temporal frame':
                list_results_base.append(dict_names[new_dict[key]])
            elif diagnosis_type[j] == 'mixed':
                list_results_proposed.append(dict_names[new_dict[key]])

        list_results_base = [unique_vals.index(x) for x in list_results_base]
        list_results_proposed = [unique_vals.index(x) for x in list_results_proposed]

        acc_base = accuracy_score(list_results_base, real_vals_base)
        acc_prop = accuracy_score(list_results_proposed, real_vals_prop)

        prec_base = precision_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        prec_prop = precision_score(list_results_proposed, real_vals_prop, average='macro', zero_division=1)

        rec_base = recall_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        rec_prop = recall_score(list_results_proposed, real_vals_prop, average='macro', zero_division=1)

        f1_base = f1_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        f1_prop = f1_score(list_results_proposed, real_vals_prop, average='macro', zero_division=1)

        mathews_base = matthews_corrcoef(list_results_base, real_vals_base)
        mathews_prop = matthews_corrcoef(list_results_proposed, real_vals_prop)

        print(f'acc base: {acc_base}, acc prop: {acc_prop}')
        print(f'prec base: {prec_base}, prec prop: {prec_prop}')
        print(f'rec base: {rec_base}, rec prop: {rec_prop}')
        print(f'f1 base: {f1_base}, f1 prop: {f1_prop}')
        print(f'Mathews CC base: {mathews_base}, Mathews CC prop: {mathews_prop}')

        list_acc_base.append(acc_base)
        list_prec_base.append(prec_base)
        list_rec_base.append(rec_base)
        list_f1_base.append(f1_base)
        list_mathews_base.append(mathews_base)

        list_acc_prop.append(acc_prop)
        list_prec_prop.append(prec_prop)
        list_rec_prop.append(rec_prop)
        list_f1_prop.append(f1_prop)
        list_mathews_prop.append(mathews_prop)

    print(f'Median acc base: {np.median(list_acc_base)} +- {np.std(list_acc_base)}, '
          f'median acc prop : {np.median(list_acc_prop)} +- {np.std(list_acc_prop)} '
          f' Mann-Whitney U test{scipy.stats.mannwhitneyu(list_acc_base, list_acc_prop)}')

    print(f'Median prec base: {np.median(list_prec_base)} +- {np.std(list_prec_base)}, '
          f'median prec prop : {np.median(list_prec_prop)} +- {np.std(list_prec_prop)}'
          f'Mann-Whitney U test{scipy.stats.mannwhitneyu(list_prec_base, list_prec_prop)}')

    print(f'Median rec base: {np.median(list_rec_base)} +- {np.std(list_rec_base)}, '
          f'median rec prop : {np.median(list_rec_prop)} +- {np.std(list_rec_prop)}'
          f'Mann-Whitney U test{scipy.stats.mannwhitneyu(list_rec_base, list_rec_prop)}')

    print(f'Median f-1 base: {np.median(list_f1_base)} +- {np.std(list_f1_base)}, '
          f'median f-1 prop : {np.median(list_f1_prop)} +- {np.std(list_f1_prop)}'
          f'Mann-Whitney U test{scipy.stats.mannwhitneyu(list_f1_base, list_f1_prop)}')

    print(f'Median Matthews CC base: {np.median(list_mathews_base)} +- {np.std(list_mathews_base)}, '
          f'median Matthews CC prop : {np.median(list_mathews_prop)} +- {np.std(list_mathews_prop)}'
          f'Mann-Whitney U test{scipy.stats.mannwhitneyu(list_mathews_base, list_mathews_prop)}')

    dictionary_results = {'acc base': list_acc_base, 'acc proposed': list_acc_prop,
                          'prec base': list_prec_base, 'prec porposed': list_prec_prop,
                          'rec base': list_rec_base, 'rec proposed': list_rec_prop,
                          'f-1 base': list_f1_base, 'f-1 proposed': list_f1_prop,
                          'mathews CC base': list_mathews_base, 'mathews CC prop': list_mathews_prop}

    boxplot_users_experiment(dictionary_results)


def compare_diagnosis_usrs(df_real_vals, df_results):
    dict_names = {'Low Grade': 'LGC',
                  'Non-Tumor Lesion': 'NTL',
                  'Non-Suspicious Tissue': 'HLT',
                  'High Grade': 'HGC'}

    list_acc_base = list()
    list_prec_base = list()
    list_rec_base = list()
    list_f1_base = list()
    list_mathews_base = list()
    list_kohen_base = list()


    real_vals = list()
    diagnosis_type = list()
    keys_of_interest = list(range(21, 60))
    question_num = df_real_vals['Question'].tolist()
    type_class = df_real_vals['REAL CLASS'].tolist()
    data_type = df_real_vals['Type'].tolist()

    for j, num in enumerate(question_num):
        if not np.isnan(num):
            if int(num) in keys_of_interest:
                real_vals.append(type_class[j])
                diagnosis_type.append(data_type[j])

    unique_vals = list(np.unique(real_vals))
    real_vals = [unique_vals.index(x) for x in real_vals]
    real_vals_prop = [x for j, x in enumerate(real_vals) if diagnosis_type[j] == 'mixed']
    real_vals_base = [x for j, x in enumerate(real_vals) if diagnosis_type[j] == 'temporal frame']

    keys_of_interest = ['Q'+str(j)+'.' for j in keys_of_interest]

    for j in range(len(df_results['Timestamp'].tolist())):
        list_results_base = list()
        list_results_proposed = list()
        case_dict = df_results.iloc[j].to_dict()
        new_dict = {k.replace(' ', ''):v for k, v in case_dict.items()}
        new_dict = {k.replace('Pleaseselecttheimagethatlooksmorerealistic', ''):v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythattheimagescorrespondto.', ''): v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythatyouthinktheimagesbelongsto.', ''): v for k, v in new_dict.items()}
        for j, key in enumerate(keys_of_interest):
            if diagnosis_type[j] == 'temporal frame':
                list_results_base.append(dict_names[new_dict[key]])
            elif diagnosis_type[j] == 'mixed':
                list_results_proposed.append(dict_names[new_dict[key]])

        list_results_base = [unique_vals.index(x) for x in list_results_base]
        list_results_proposed = [unique_vals.index(x) for x in list_results_proposed]

        acc_base = accuracy_score(list_results_base, real_vals_base)
        prec_base = precision_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        rec_base = recall_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        f1_base = f1_score(list_results_base, real_vals_base, average='macro', zero_division=1)
        mathews_base = matthews_corrcoef(list_results_base, real_vals_base)
        cohen_kappa = cohen_kappa_score(list_results_base, real_vals_base)

        list_acc_base.append(acc_base)
        list_prec_base.append(prec_base)
        list_rec_base.append(rec_base)
        list_f1_base.append(f1_base)
        list_mathews_base.append(mathews_base)
        list_kohen_base.append(cohen_kappa)

    dictionary_performance = {'name model': 'users'}
    dictionary_performance['domain translation'] = '-'
    dictionary_performance['unlabel data'] = '-'
    dictionary_performance['test data'] = 'ALL'
    dictionary_performance['Accuracy'] = str(round(np.median(list_acc_base), 3)) + ' pm ' + str(round(np.std(list_acc_base), 3))
    dictionary_performance['p-val Accuracy'] = '-'
    dictionary_performance['Precision'] = str(round(np.median(list_prec_base), 3)) + ' pm ' + str(round(np.std(list_prec_base), 3))
    dictionary_performance['p-val Precision'] = '-'
    dictionary_performance['Recall'] = str(round(np.median(list_rec_base), 3)) + ' pm ' + str(round(np.std(list_rec_base), 3))
    dictionary_performance['p-val Recall'] = '-'
    dictionary_performance['F-1'] = str(round(np.median(list_f1_base), 3)) + ' pm ' + str(round(np.std(list_f1_base), 3))
    dictionary_performance['p-val F-1'] = '-'
    dictionary_performance['Matthews CC'] = str(round(np.median(list_mathews_base), 3)) + ' pm ' + str(round(np.std(list_mathews_base), 3))
    dictionary_performance['p-val Matthews CC'] = '-'
    dictionary_performance['Cohen'] = str(round(np.median(list_kohen_base), 3)) + ' pm ' + str(round(np.std(list_kohen_base), 3))
    dictionary_performance['p-val Cohen'] = '-'

    return dictionary_performance


def calc_average_performances(data_model_1, data_baseline=None, metric_list=None, comparison_attribute='name_model',
                              training_data='ALL', training_data_baseline='WLI', domain_translation=False,
                              unlabel_data=False):

    def print_p_val(p_val):
        if p_val < 0.0001:
            print_val = '<0.0001'
        elif p_val < 0.001:
            print_val = '<0.001'
        elif p_val < 0.01:
            print_val = '<0.01'
        elif p_val < 0.05:
            print_val = '<0.05'
        else:
            print_val = '>0.05'
        return print_val

    name_1 = list(np.unique(data_model_1[comparison_attribute].tolist()))[-1]
    dictionary_performance = {'name model': name_1}
    data_subset = metric_list[0].split(' ')[-1]
    dictionary_performance['domain translation'] = domain_translation
    dictionary_performance['unlabel data'] = unlabel_data
    dictionary_performance['test data'] = data_subset
    for metric in metric_list:
        if 'Matthews CC' in metric:
            metric_name = 'Matthews CC'
        else:
            metric_name = metric.split(' ')[0]

        df1 = data_model_1.loc[data_model_1['training_data_used'] == training_data]
        list_metric_model_1 = df1[metric].tolist()
        dictionary_performance[metric_name] = str(round(np.median(list_metric_model_1), 3)) + ' pm ' + str(round(np.std(list_metric_model_1), 3))
        if data_baseline is None:
            dictionary_performance['p-val ' + metric_name] = '-'

        else:
            df2 = data_baseline.loc[data_baseline['training_data_used'] == training_data_baseline]
            list_metric_model_2 = df2[metric].tolist()
            result_mwu_test = scipy.stats.mannwhitneyu(list_metric_model_1, list_metric_model_2)[1]
            dictionary_performance['p-val ' + metric_name] = round(result_mwu_test, 3)#print_p_val(result_mwu_test)

    return dictionary_performance


def extract_data_to_list(data_model_1, metric_list=None, comparison_attribute='name_model'):
    list_values = list()
    output_dict = {}
    name_1 = list(np.unique(data_model_1[comparison_attribute].tolist()))[-1]
    training_data_used = list(np.unique(data_model_1['training_data_used'].tolist()))
    for metric in metric_list:
        for training_data in training_data_used:
            df1 = data_model_1.loc[data_model_1['training_data_used'] == training_data]
            list_metric_model_1 = df1[metric].tolist()
            list_values.append(list_metric_model_1)
            output_dict[metric] = list_metric_model_1

    return output_dict


def Mann_Whitney_U_test(metrics_models, metrics, unique_models=None, analysis_type='by_model'):
    if analysis_type == 'by_model':

        print(metrics_models)
        r1 = scipy.stats.mannwhitneyu(metrics_models['densenet'][metrics[0]], metrics_models['resnet101'][metrics[0]])
        r2 = scipy.stats.mannwhitneyu(metrics_models['densenet'][metrics[1]], metrics_models['resnet101'][metrics[1]])
        r3 = scipy.stats.mannwhitneyu(metrics_models['densenet'][metrics[2]], metrics_models['resnet101'][metrics[2]])
        print('densenet vs resnet101')
        print(metrics[0], r1)
        print(metrics[1], r2)
        print(metrics[2], r3)
        jf_model = [m for m in unique_models if 'jf' in m].pop()
        sf_model = [m for m in unique_models if 'sf' in m].pop()

        r4 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[0]], metrics_models['resnet101'][metrics[0]])
        r5 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[1]], metrics_models['resnet101'][metrics[1]])
        r6 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[2]], metrics_models['resnet101'][metrics[2]])
        print(f'{jf_model} vs resnet')
        print(metrics[0], r4)
        print(metrics[1], r5)
        print(metrics[2], r6)
        r7 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[0]], metrics_models['resnet101'][metrics[0]])
        r8 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[1]], metrics_models['resnet101'][metrics[1]])
        r9 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[2]], metrics_models['resnet101'][metrics[2]])
        print(f'{sf_model} vs resnet')
        print(metrics[0], r7)
        print(metrics[1], r8)
        print(metrics[2], r9)

        r10 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[0]], metrics_models['densenet'][metrics[0]])
        r11 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[1]], metrics_models['densenet'][metrics[1]])
        r12 = scipy.stats.mannwhitneyu(metrics_models[jf_model][metrics[2]], metrics_models['densenet'][metrics[2]])
        print(f'{jf_model} vs densenet')
        print(metrics[0], r10)
        print(metrics[1], r11)
        print(metrics[2], r12)
        r13 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[0]], metrics_models['densenet'][metrics[0]])
        r14 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[1]], metrics_models['densenet'][metrics[1]])
        r15 = scipy.stats.mannwhitneyu(metrics_models[sf_model][metrics[2]], metrics_models['densenet'][metrics[2]])
        print(f'{sf_model} vs densenet')
        print(metrics[0], r13)
        print(metrics[1], r14)
        print(metrics[2], r15)
    elif analysis_type == 'by_data_type':
        pass


def generate_dctionary_metrics_results(dataframe, metrics, domain_type='ALL'):
    list_models = dataframe['name model'].tolist()
    unique_models = list(np.unique(list_models))
    metrics_models = {m: {metric: list() for metric in metrics[:]} for m in unique_models}
    for i, model in enumerate(list_models):
        row = dataframe.loc[i]
        if row['training_data_used'] == domain_type:
            name_model = row['name model']
            metrics_models[name_model][metrics[0]].append(row[metrics[0]])
            metrics_models[name_model][metrics[1]].append(row[metrics[1]])
            metrics_models[name_model][metrics[2]].append(row[metrics[2]])

    return metrics_models, unique_models


def generate_data_list_plot(metrics_models, metrics):
    data = list()
    for name_model in metrics_models:
        data.append(metrics_models[name_model][metrics[0]])
        data.append(metrics_models[name_model][metrics[1]])
        data.append(metrics_models[name_model][metrics[2]])

    return data


def box_plot_matplotlib(dataframe, title='', y_label='', computer_stat_sig=True, metrics=None,
                        analysis_type='by_model'):
    # Generate some random indices that we'll use to resample the original data
    # arrays. For code brevity, just use the same random indices for each array

    if not metrics:
        metrics = list(dataframe.columns.values)
        metrics.remove('name model')
        if 'training_data_used' in metrics:
            metrics.remove('training_data_used')

    # you need to group the results by model name
    if analysis_type == 'by_model':

        metrics_models, unique_models = generate_dctionary_metrics_results(dataframe, metrics)
        data = generate_data_list_plot(metrics_models, metrics)

        if computer_stat_sig:
            Mann_Whitney_U_test(metrics_models, metrics, unique_models, analysis_type=analysis_type)

        labels = ['' for _ in range(len(data))]
        for i, label in enumerate(labels):
            if i % 3 == 0:
                labels[i + 1] = unique_models[int(i/3)]

    elif analysis_type == 'by_data_type':
        list_data_used = dataframe['training_data_used'].tolist()
        unique_types_data = np.unique(list_data_used)
        data = list()
        metrics_models = {}
        unique_models = list()
        for type_training_data in unique_types_data:
            metric, uniques = generate_dctionary_metrics_results(dataframe, metrics,
                                                                  domain_type=type_training_data)
            data = data + generate_data_list_plot(metrics_models, metrics)
            metrics_models = {**metrics_models, **metric}
            unique_models = unique_models + uniques
        print(metrics_models)
        if computer_stat_sig:
            Mann_Whitney_U_test(metrics_models, metrics, unique_models, analysis_type=analysis_type)

        labels = ['' for x in range(len(data))]
        for i, label in enumerate(labels):
            if i % 3 == 0:
                labels[i + 1] = unique_models[int(i/3)]

    fig, ax1 = plt.subplots(figsize=(15, 9))
    fig.canvas.set_window_title('Boxplot Comparison')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.85, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    # ax1.set_title(Title)
    # ax1.set_xlabel('Model')
    ax1.set_ylabel(y_label, fontsize=15)

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    box_colors = ['green', 'royalblue', 'orange']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    averages = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 3]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        averages[i] = np.average(data[i])
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.1
    bottom = -0.1
    ax1.set_ylim(bottom, top)

    ax1.set_xticklabels(labels, fontsize=15, weight='bold')

    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 3
        ax1.text(pos[tick], 0.95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='large',
                 weight=weights[k], color=box_colors[k])

    # Finally, add a basic legend

    fig.text(0.10, 0.15, 'All data',
             backgroundcolor=box_colors[0], color='black', weight='roman',
             size='large')
    fig.text(0.10, 0.12, 'WLI data',
             backgroundcolor=box_colors[1],
             color='black', weight='roman', size='large')
    fig.text(0.10, 0.09, 'NBI data',
             backgroundcolor=box_colors[2],
             color='black', weight='roman', size='large')

    fig.text(0.10, 0.05, '*', color='white', backgroundcolor='silver',
             weight='roman', size='large')
    fig.text(0.115, 0.05, 'Average Value', color='black', weight='roman',
             size='large')
    plt.show()


def boxplot_users_experiment(data_dictionary):
    data = [data_dictionary[x] for x in data_dictionary]
    labels = data_dictionary.keys()
    fig, ax1 = plt.subplots(figsize=(15, 9))
    fig.canvas.set_window_title('Boxplot Comparison')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.85, bottom=0.25)

    bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    # ax1.set_title(Title)
    # ax1.set_xlabel('Model')
    y_label = ''
    ax1.set_ylabel(y_label, fontsize=15)

    # Now fill the boxes with desired colors
    # box_colors = ['darkkhaki', 'royalblue']
    box_colors = ['royalblue', 'darkorange']
    num_boxes = len(data)
    medians = np.empty(num_boxes)
    averages = np.empty(num_boxes)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between Dark Khaki and Royal Blue
        ax1.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        # Now draw the median lines back over what we just filled in
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            ax1.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
        averages[i] = np.average(data[i])
        # Finally, overplot the sample averages, with horizontal alignment
        # in the center of each box
        ax1.plot(np.average(med.get_xdata()), np.average(data[i]),
                 color='w', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, num_boxes + 0.5)
    top = 1.1
    bottom = -0.1
    ax1.set_ylim(bottom, top)

    ax1.set_xticklabels(labels, fontsize=15, weight='bold')

    pos = np.arange(num_boxes) + 1
    upper_labels = [str(round(s, 2)) for s in medians]
    weights = ['bold', 'semibold', 'semibold']
    for tick, label in zip(range(num_boxes), ax1.get_xticklabels()):
        k = tick % 2
        ax1.text(pos[tick], 0.95, upper_labels[tick],
                 transform=ax1.get_xaxis_transform(),
                 horizontalalignment='center', size='large',
                 weight=weights[k], color=box_colors[k])

    plt.show()


def calculate_p_values(data_frame, selection, metrics, selected_values=None, base_model='resnet101', sub_selection=None):
    print('selection', data_frame[selection].unique().tolist())
    print('sub_selection:', sub_selection)
    # selecting rows based on condition

    base_model_df = data_frame.loc[data_frame[selection] == base_model]
    base_metric_all = base_model_df[metrics[0]].tolist()
    base_metric_wli = base_model_df[metrics[1]].tolist()
    base_metric_nbi = base_model_df[metrics[2]].tolist()
    base_metrics = [base_model_df[metrics[j]].tolist() for j in range(len(metrics))]
    options_sub_selection = data_frame[sub_selection].unique().tolist()
    if selected_values:
        for value in selected_values:

            if sub_selection:
                for condition in options_sub_selection:
                    rslt_df = data_frame[(data_frame[selection] == value) & (data_frame[sub_selection] == condition)]
                    for j, metric in enumerate(metrics):
                        print(f'{value}: {metric} in {condition} data')
                        metric_vals = rslt_df[metric].tolist()
                        print(f'mean: {np.mean(metric_vals)}, median: {np.median(metric_vals)}, '
                              f'+- {np.std(metric_vals)}')
                        mu = scipy.stats.mannwhitneyu(base_metrics[j], metric_vals)
                        print(f'p-val {value} vs {base_model}', )
                        print(mu)
            else:
                rslt_df = data_frame.loc[data_frame[selection] == value]

            for j, metric in enumerate(metrics):
                print(f'{value}: {metric}')
                metric_vals = rslt_df[metric].tolist()
                print(f'mean: {np.mean(metric_vals)}, median: {np.median(metric_vals)}, +- {np.std(metric_vals)}')
                mu = scipy.stats.mannwhitneyu(base_metrics[j], metric_vals)
                print(f'p-val {value} vs {base_model}', )
                print(mu)
    print('selection', selection)


def boxplot_seaborn(data_frame, x_axis, y_axis, title_plot='', hue=None, order=None):
    meanprops = {"marker": "*",
                 "markerfacecolor": "white",
                 "markeredgecolor": "black",
                 "markersize": "10"}
    fig1 = plt.figure(1, figsize=(11, 7))
    fig1.canvas.set_window_title(title_plot)
    fig1.suptitle(title_plot, fontsize=14)
    ax1 = fig1.add_subplot(131)
    if hue:
        ax1 = sns.stripplot(x=x_axis, y=y_axis[0], hue=hue,
                      data=data_frame, jitter=True,
                      split=True, linewidth=0.3, size=3, order=order)
        ax1 = sns.boxplot(x=x_axis, y=y_axis[0], data=data_frame, showmeans=True,
                          hue=hue, meanprops=meanprops, order=order)
        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax1.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        ax1.get_legend().remove()

        #l = plt.legend(handles[-2:], labels[-2:], title="Training data used",)

    else:
        ax1 = sns.boxplot(x=x_axis, y=y_axis[0], data=data_frame, showmeans=True,
                          meanprops=meanprops, order=order)
        ax1 = sns.swarmplot(x=x_axis, y=y_axis[0], data=data_frame, color=".25", order=order)

    metric = y_axis[0].split(' ')
    ax1.set_ylabel(metric[0])
    ax1.set_ylim([0, 1.05])
    ax1.title.set_text('ALL')

    ax2 = fig1.add_subplot(132)
    if hue:
        ax2 = sns.stripplot(x=x_axis, y=y_axis[1], hue=hue,
                            data=data_frame, jitter=True,
                            split=True, linewidth=0.3, size=3, order=order)
        ax2 = sns.boxplot(x=x_axis, y=y_axis[1], data=data_frame, showmeans=True,
                          hue=hue, meanprops=meanprops, order=order)
        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax2.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        #l = plt.legend(handles[-2:], labels[-2:], title="Training data used",)
        ax2.get_legend().remove()
    else:
        ax2 = sns.boxplot(x=x_axis, y=y_axis[1], data=data_frame, showmeans=True,
                          meanprops=meanprops, order=order)
        ax2 = sns.swarmplot(x=x_axis, y=y_axis[1], data=data_frame, color=".25", order=order)

    ax2.set_ylim([0, 1.05])
    ax2.title.set_text('WLI')
    ax2.axes.get_yaxis().set_visible(False)

    ax3 = fig1.add_subplot(133)
    if hue:
        ax3 = sns.stripplot(x=x_axis, y=y_axis[2], hue=hue,
                            data=data_frame, jitter=True,
                            split=True, linewidth=0.3, size=3, order=order)
        ax3 = sns.boxplot(x=x_axis, y=y_axis[2], data=data_frame, showmeans=True,
                          hue=hue, meanprops=meanprops, order=order)
        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax3.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        l = plt.legend(handles[-2:], labels[-2:], title="Training data used")
    else:
        ax3 = sns.boxplot(x=x_axis, y=y_axis[2], data=data_frame, showmeans=True,
                          hue=hue, order=order)
        ax3 = sns.swarmplot(x=x_axis, y=y_axis[2], data=data_frame, color=".25",
                            order=order)


    ax3.set_ylim([0, 1.05])
    ax3.title.set_text('NBI')
    ax3.axes.get_yaxis().set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
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
