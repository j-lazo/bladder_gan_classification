import shutil
from absl import app, flags
from absl.flags import FLAGS
import os
import tqdm
import numpy as np
import pandas as pd
from utils import data_analysis as daa
from utils.data_management import file_management as fam
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import scipy


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


def compare_datasets_v1(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'Accuracy'
    df = pd.read_csv(dir_to_csv)

    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    # 'bladder_tissue_classification_v3_augmented',
    # 'bladder_tissue_classification_v3_gan_new',]

    chosen_trained_data = ['WLI', 'ALL']
    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,
                            'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi',
                            'temp_not_complete_wli2nbi_ssim']

    dictionary_selection_resnet_bv3 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v3'],
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   'date':['01-07-2022', '20-06-2022']}
    selection_resnet_bv3 = select_specific_cases(df, dictionary_selection_resnet_bv3)
    dictionary_selection_resnet_bv3_t = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v3'],
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   'date':['26-07-2022']}
    selection_resnet_bv3_t = select_specific_cases(df, dictionary_selection_resnet_bv3_t)

    dictionary_selection_resnet_bv4 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v4'],
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   }
    selection_resnet_bv4 = select_specific_cases(df, dictionary_selection_resnet_bv4)

    dictionary_selection_resnet_bv4_1 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes,
                                       'dataset': ['bladder_tissue_classification_v4_1'],
                                       'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                       }
    selection_resnet_bv4_1 = select_specific_cases(df, dictionary_selection_resnet_bv4_1)

    dictionary_selection_resnet_bv4_2 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes,
                                       'dataset': ['bladder_tissue_classification_v4_2'],
                                       'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                       }
    selection_resnet_bv4_2 = select_specific_cases(df, dictionary_selection_resnet_bv4_2)

    dictionary_selection_resnet_bv5 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                         'batch_size': chosen_batch_sizes,
                                         'dataset': ['bladder_tissue_classification_v5'],
                                         'backbone GAN': chosen_gan_backbones,
                                         'training_data_used': chosen_trained_data,
                                         }
    selection_resnet_bv5 = select_specific_cases(df, dictionary_selection_resnet_bv5)

    dictionary_selection_resnet_bv6 = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                         'batch_size': chosen_batch_sizes,
                                         'dataset': ['bladder_tissue_classification_v6'],
                                         'backbone GAN': chosen_gan_backbones,
                                         'training_data_used': chosen_trained_data,
                                         }

    selection_resnet_bv6 = select_specific_cases(df, dictionary_selection_resnet_bv6)

    dictionary_selection_separat_bv3 = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes,
                                       'dataset': ['bladder_tissue_classification_v3'],
                                       'backbone GAN': ['general_wli2nbi'], 'training_data_used': chosen_trained_data,
                                       'date':['06-07-2022']}
    selection_separat_bv3 = select_specific_cases(df, dictionary_selection_separat_bv3)

    dictionary_selection_separat_bv4 = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes,
                                       'dataset': ['bladder_tissue_classification_v4'],
                                       'backbone GAN': ['general_wli2nbi'], 'training_data_used': chosen_trained_data,
                                       }
    selection_separat_bv4 = select_specific_cases(df, dictionary_selection_separat_bv4)
    dictionary_selection_separat_bv4_1 = {'name_model': ['gan_model_separate_features'],
                                        'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes,
                                        'dataset': ['bladder_tissue_classification_v4_1'],
                                        'backbone GAN': ['general_wli2nbi'], 'training_data_used': chosen_trained_data,
                                        }
    selection_separat_bv4_1 = select_specific_cases(df, dictionary_selection_separat_bv4_1)
    dictionary_selection_separat_bv3_t = {'name_model': ['gan_model_separate_features'],
                                        'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes,
                                        'dataset': ['bladder_tissue_classification_v3'],
                                        'backbone GAN': ['general_wli2nbi'], 'training_data_used': chosen_trained_data,
                                        'date': ['26-07-2022']}
    selection_separat_bv3_t = select_specific_cases(df, dictionary_selection_separat_bv3_t)

    dictionary_selection_separat_bv5 = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes,
                                       'dataset': ['bladder_tissue_classification_v5'],
                                       'backbone GAN': ['not_complete_wli2nbi'], 'training_data_used': chosen_trained_data,
                                       }
    selection_separat_bv5 = select_specific_cases(df, dictionary_selection_separat_bv5)

    dictionary_selection_separat_bv6 = {'name_model': ['gan_model_separate_features'],
                                        'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes,
                                        'dataset': ['bladder_tissue_classification_v6'],
                                        'backbone GAN': ['not_complete_wli2nbi'],
                                        'training_data_used': chosen_trained_data,
                                        }
    selection_separat_bv6 = select_specific_cases(df, dictionary_selection_separat_bv6)
    dictionary_selection_separat_bv4_2 = {'name_model': ['gan_model_separate_features'],
                                        'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes,
                                        'dataset': ['bladder_tissue_classification_v4_2'],
                                        'backbone GAN': ['not_complete_wli2nbi'],
                                        'training_data_used': chosen_trained_data,
                                        }
    selection_separat_bv4_2 = select_specific_cases(df, dictionary_selection_separat_bv4_2)
    print(selection_separat_bv4_2)
    selection_resnet_bv3.loc[selection_resnet_bv3["name_model"] == "resnet101", "name_model"] = 'B1'
    selection_resnet_bv4.loc[selection_resnet_bv4["name_model"] == "resnet101", "name_model"] = 'B2'
    selection_resnet_bv4_1.loc[selection_resnet_bv4_1["name_model"] == "resnet101", "name_model"] = 'B3'
    selection_resnet_bv3_t.loc[selection_resnet_bv3_t["name_model"] == "resnet101", "name_model"] = 'B1-R'
    selection_resnet_bv4_2.loc[selection_resnet_bv4_2["name_model"] == "resnet101", "name_model"] = 'B2-C'
    selection_resnet_bv5.loc[selection_resnet_bv5["name_model"] == "resnet101", "name_model"] = 'F2'
    selection_resnet_bv6.loc[selection_resnet_bv6["name_model"] == "resnet101", "name_model"] = 'F3'


    selection_separat_bv3.loc[selection_separat_bv3["name_model"] == "gan_model_separate_features", "name_model"] = 'P1'
    selection_separat_bv4.loc[selection_separat_bv4["name_model"] == "gan_model_separate_features", "name_model"] = 'P2'
    selection_separat_bv4_1.loc[selection_separat_bv4_1["name_model"] == "gan_model_separate_features", "name_model"] = 'P3'
    selection_separat_bv3_t.loc[selection_separat_bv3_t["name_model"] == "gan_model_separate_features", "name_model"] = 'P1-R'
    selection_separat_bv4_2.loc[selection_separat_bv4_2["name_model"] == "gan_model_separate_features", "name_model"] = 'P1-C'
    selection_separat_bv5.loc[selection_separat_bv5["name_model"] == "gan_model_separate_features", "name_model"] = 'P-F2'
    selection_separat_bv6.loc[selection_separat_bv6["name_model"] == "gan_model_separate_features", "name_model"] = 'P-F3'


    selection = pd.concat([selection_resnet_bv3,
                           selection_resnet_bv4,
                           selection_resnet_bv4_1,
                           selection_resnet_bv3_t,
                           selection_resnet_bv4_2,
                           selection_resnet_bv5,
                           selection_resnet_bv6,
                           selection_separat_bv3,
                           selection_separat_bv4,
                           selection_separat_bv4_1,
                           selection_separat_bv3_t,
                           selection_separat_bv4_2,
                           selection_separat_bv5,
                           selection_separat_bv6])

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    x_axis = 'name_model'
    y_axis = metrics_box
    title_plot = 'Comparison Datasets'  # name_model.replace('_', ' ')
    # daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')


def fid_score_vs_metrics(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                         metrics='all', exclusion_criteria=None):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'F-1'
    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    chosen_dataset = ['bladder_tissue_classification_v3']
    chosen_trained_data = ['ALL']
    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    results_dir = os.path.join(os.getcwd(), 'results')
    df = pd.read_csv(dir_to_csv)
    yaml_files = [f for f in os.listdir(results_dir) if f.endswith('yaml')]

    list_dif_results = list()

    for file in yaml_files:
        yaml_file_dir = os.path.join(results_dir, file)
        list_dif_results.append(fam.read_yaml_file(yaml_file_dir))

    x_all = list()
    y_all = list()
    y_all_stdv = list()

    x_wli = list()
    y_wli = list()
    y_wli_stdv = list()

    x_nbi = list()
    y_nbi = list()
    y_nbi_stdv = list()

    list_names = list()

    exlcusion_list = []
    #exlcusion_list = ['fine_tune_2C1', 'fine_tune_2C2', 'fine_tune']
    for dictionary in list_dif_results:
        gan_selection = dictionary['GAN model']
        if gan_selection not in exlcusion_list:
            list_names.append(gan_selection)
            general_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                      'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                      'backbone GAN': [gan_selection], 'training_data_used': chosen_trained_data,
                                      }
            df_selection = select_specific_cases(df, general_selection_dict)
            dicta = daa.extract_data_to_list(df_selection, metric_list=metrics_box,
                                             comparison_attribute='backbone GAN')

            #y_all.append(dicta['Accuracy ALL'])
            #y_wli.append(dicta['Accuracy WLI'])
            #y_nbi.append(dicta['Accuracy NBI'])
            if gan_selection == 'not_complete_wli2nbi':
                y_all.append(np.mean(dicta[metrics_box[0]])-0.00)
                y_wli.append(np.mean(dicta[metrics_box[1]])-0.00)
                y_nbi.append(np.mean(dicta[metrics_box[2]])-0.00)
            else:
                y_all.append(np.mean(dicta[metrics_box[0]]))
                y_wli.append(np.mean(dicta[metrics_box[1]]))
                y_nbi.append(np.mean(dicta[metrics_box[2]]))

            y_all_stdv.append(np.std(dicta[metrics_box[0]])/2)
            y_wli_stdv.append(np.std(dicta[metrics_box[1]])/2)
            y_nbi_stdv.append(np.std(dicta[metrics_box[2]])/2)

            x_all.append(dictionary['FID ALL'])
            x_wli.append(dictionary['FID WLI'])
            x_nbi.append(dictionary['FID NBI'])

    z_wli = [x for _, x, _ in sorted(zip(x_wli, y_wli, y_wli_stdv))]
    z_nbi = [x for _, x, _ in sorted(zip(x_nbi, y_nbi, y_nbi_stdv))]
    z_all = [x for _, x, _ in sorted(zip(x_all, y_all, y_all_stdv))]

    e_bar_all = [x for _, _, x in sorted(zip(x_all, y_all, y_all_stdv))]
    e_bar_nbi = [x for _, _, x in sorted(zip(x_nbi, y_nbi, y_nbi_stdv))]
    e_bar_wli = [x for _, _, x in sorted(zip(x_wli, y_wli, y_wli_stdv))]

    z_names_wli = [x for _, x in sorted(zip(x_wli, list_names))]
    z_names_nbi = [x for _, x in sorted(zip(x_nbi, list_names))]
    dict_show_wli = {}
    for j, x in enumerate(sorted(x_wli)):
        dict_show_wli[z_names_wli[j]] = x
    print(dict_show_wli)
    dict_show_nbi = {}
    for j, x in enumerate(sorted(x_nbi)):
        dict_show_nbi[z_names_nbi[j]] = x
    print(dict_show_nbi)

    plt.figure()
    #plt.plot(sorted(x_all), z_all, '-o', label='all')
    plt.errorbar(sorted(x_wli), z_wli, yerr=e_bar_wli, uplims=True, lolims=True, label='$G_{A2B}$')
    plt.errorbar(sorted(x_nbi), z_nbi, yerr=e_bar_nbi, uplims=True, lolims=True, label='$G_{B2A}$')

    """z_names_nbi = z_names_nbi[::-1]
    z_names_wli = z_names_wli[::-1]

    for j, zipped in enumerate(zip(x_wli, z_wli)):
        x = zipped[0]
        y = zipped[1]
        label = z_names_wli[j]
        plt.annotate(label,  # this is the value which we want to label (text)
                     (x, y),  # x and y is the points location where we have to label
                     textcoords="offset points",
                     xytext=(0, 10),  # this for the distance between the points
                     # and the text label
                     ha='center',
                     arrowprops=dict(arrowstyle="->", color='green'))

    for j, zipped in enumerate(zip(x_nbi, z_nbi)):
        x = zipped[0]
        y = zipped[1]
        label = z_names_nbi[j]
        plt.annotate(label,  # this is the value which we want to label (text)
                     (x, y),  # x and y is the points location where we have to label
                     textcoords="offset points",
                     xytext=(0, 10),  # this for the distance between the points
                     # and the text label
                     ha='center',
                     arrowprops=dict(arrowstyle="->", color='green'))"""

    plt.xlabel('FID Score')
    plt.ylabel('Avg.' + metric_analysis)
    plt.legend(loc='best')



    plt.show()


def compare_gans_boxplots(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'Accuracy'
    df = pd.read_csv(dir_to_csv)

    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    chosen_dataset = ['bladder_tissue_classification_v3']
                      #'bladder_tissue_classification_v3_augmented',
                      #'bladder_tissue_classification_v3_gan_new',]

    chosen_trained_data = ['WLI', 'ALL']
    dates_selection = ['21-06-2022', '22-06-2022']
    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,
                            'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi',
                            'temp_not_complete_wli2nbi_ssim']

    dictionary_selection_resnet = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   }
    selection_resnet = select_specific_cases(df, dictionary_selection_resnet)
    dict_replace = {x: 'A' for x in np.unique(selection_resnet['backbone GAN'].tolist())}
    dict_replace[np.nan] = 'A'
    selection_resnet['backbone GAN'] = selection_resnet['backbone GAN'].map(dict_replace, na_action=None)

    base_selection_dictionary = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': ['not_complete_wli2nbi'], 'training_data_used': chosen_trained_data,
                            'date':['21-06-2022', '22-06-2022', '06-07-2022', '05-07-2022']
                                 }
    base_selection = select_specific_cases(df, base_selection_dictionary)

    general_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': ['general_wli2nbi'], 'training_data_used': chosen_trained_data,
                            }
    general_selection = select_specific_cases(df, general_selection_dict)

    general_ssim_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['general_wli2nbi_ssim'], 'training_data_used': chosen_trained_data,
                              }
    general_ssim_selection = select_specific_cases(df, general_ssim_selection_dict)

    not_complete_ssim_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['not_complete_wli2nbi_ssim'], 'training_data_used': chosen_trained_data,
                              }
    not_complete_ssim_selection = select_specific_cases(df, not_complete_ssim_selection_dict)
    not_complete_ssim_selection_dict_2 = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['not_complete_wli2nbi_ssim_2'], 'training_data_used': chosen_trained_data,
                              }
    not_complete_ssim_selection_2 = select_specific_cases(df, not_complete_ssim_selection_dict_2)
    temp_not_complete_ssim_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['temp_not_complete_wli2nbi_ssim'], 'training_data_used': chosen_trained_data,
                              }
    temp_not_complete_ssim_selection = select_specific_cases(df, temp_not_complete_ssim_selection_dict)

    dict_selection_wli2nbi = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['wli2nbi'], 'training_data_used': chosen_trained_data,
                             }

    selection_wli2nbi = select_specific_cases(df, dict_selection_wli2nbi)

    dict_selection_wli2nbi_ssim = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                              'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                              'backbone GAN': ['wli2nbi_ssim'], 'training_data_used': chosen_trained_data,
                             }
    selection_wli2nbi_ssim = select_specific_cases(df, dict_selection_wli2nbi_ssim)

    dict_selection_wli2nbi_ssim_2 = {'name_model': ['gan_model_separate_features'],
                                   'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': ['wli2nbi_ssim_2'], 'training_data_used': chosen_trained_data,
                                   }
    selection_wli2nbi_ssim_2 = select_specific_cases(df, dict_selection_wli2nbi_ssim_2)

    dict_selection_fine_tune = {'name_model': ['gan_model_separate_features'],
                                   'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': ['fine_tune'], 'training_data_used': chosen_trained_data,
                                   }
    selection_fine_tune = select_specific_cases(df, dict_selection_fine_tune)

    dict_selection_fine_tune_2C1 = {'name_model': ['gan_model_separate_features'],
                                'learning_rate': chosen_learning_rates,
                                'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                'backbone GAN': ['fine_tune_2C1'], 'training_data_used': chosen_trained_data,
                                }
    selection_fine_tune_2C1 = select_specific_cases(df, dict_selection_fine_tune_2C1)

    dict_selection_fine_tune_2C2 = {'name_model': ['gan_model_separate_features'],
                                    'learning_rate': chosen_learning_rates,
                                    'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                    'backbone GAN': ['fine_tune_2C2'], 'training_data_used': chosen_trained_data,
                                    }
    selection_fine_tune_2C2 = select_specific_cases(df, dict_selection_fine_tune_2C2)

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]
    # compariosn base gans
    print('*********************')
    daa.compute_Mann_Whitney_U_test(base_selection, selection_wli2nbi, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(selection_wli2nbi, general_selection, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(base_selection, general_selection, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    # comparison of no ssim vs ssim
    daa.compute_Mann_Whitney_U_test(selection_wli2nbi, selection_wli2nbi_ssim_2, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(base_selection, not_complete_ssim_selection_2, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(base_selection, selection_wli2nbi_ssim_2, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    # comparison ssim base vs ssim others
    print('*********************')
    daa.compute_Mann_Whitney_U_test(selection_wli2nbi_ssim_2, not_complete_ssim_selection_2, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(not_complete_ssim_selection_2, general_ssim_selection, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(not_complete_ssim_selection_2, selection_fine_tune, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(not_complete_ssim_selection_2, selection_fine_tune_2C1, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')
    print('*********************')
    daa.compute_Mann_Whitney_U_test(not_complete_ssim_selection_2, selection_fine_tune_2C2, metric_list=metrics_box,
                                    comparison_attribute='backbone GAN')

    selection_wli2nbi.loc[selection_wli2nbi["backbone GAN"] == "wli2nbi", "backbone GAN"] = 'B'
    base_selection.loc[base_selection["backbone GAN"] == "not_complete_wli2nbi", "backbone GAN"] = 'C'
    general_selection.loc[general_selection["backbone GAN"] == "general_wli2nbi", "backbone GAN"] = 'D'
    selection_wli2nbi_ssim.loc[selection_wli2nbi_ssim["backbone GAN"] == "wli2nbi_ssim", "backbone GAN"] = 'E'
    selection_wli2nbi_ssim_2.loc[selection_wli2nbi_ssim_2["backbone GAN"] == "wli2nbi_ssim_2", "backbone GAN"] = 'X'
    not_complete_ssim_selection_2.loc[not_complete_ssim_selection_2["backbone GAN"] == "not_complete_wli2nbi_ssim_2", "backbone GAN"] = 'F'
    general_ssim_selection.loc[general_ssim_selection["backbone GAN"] == "general_wli2nbi_ssim", "backbone GAN"] = 'G'
    selection_fine_tune.loc[selection_fine_tune["backbone GAN"] == "fine_tune", "backbone GAN"] = 'FT'
    selection_fine_tune_2C1.loc[selection_fine_tune_2C1["backbone GAN"] == "fine_tune_2C1", "backbone GAN"] = 'FT1'
    selection_fine_tune_2C2.loc[selection_fine_tune_2C2["backbone GAN"] == "fine_tune_2C2", "backbone GAN"] = 'FT2'



    selection = pd.concat([selection_resnet,
                           selection_wli2nbi,
                           base_selection,
                           general_selection,
                           selection_wli2nbi_ssim,
                           selection_wli2nbi_ssim_2,
                           not_complete_ssim_selection_2,
                           general_ssim_selection,
                           selection_fine_tune,
                           selection_fine_tune_2C1,
                           selection_fine_tune_2C2,
                           ])



    x_axis = 'backbone GAN'
    y_axis = metrics_box
    title_plot = 'Comparison GANs'  # name_model.replace('_', ' ')
    # daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')


def compare_dataset_boxplots(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv', )),
                             metrics='all', exclusion_criteria=None, metric_analysis = 'Accuracy'):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    df = pd.read_csv(dir_to_csv)

    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]

    chosen_trained_data = ['WLI', 'ALL']
    dictionary_selection_resnet = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v3'],
                                   'training_data_used': chosen_trained_data,
                                   }
    selection_resnet = select_specific_cases(df, dictionary_selection_resnet)

    dict_gan_not_complete_all_converted_NBI = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes,
                                    'dataset': ['bladder_tissue_classification_v3_gan_not_complete_all+converted_NBI'],
                                    'training_data_used': chosen_trained_data,
                                               }
    selection_not_complete_all_converted_NBI = select_specific_cases(df, dict_gan_not_complete_all_converted_NBI)

    dict_temp_not_complete_ssim_all_converted_NBI= {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_temp_not_complete_ssim_all+converted_NBI'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selection_temp_not_complete_ssim_all_converted_NBI = select_specific_cases(df, dict_temp_not_complete_ssim_all_converted_NBI)
    dict_gan_not_complete_ssim_2_all_converted_NBI = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_not_complete_ssim_2_all+converted_NBI'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selection_temp_not_complete_ssim_2_all_converted_NBI = select_specific_cases(df, dict_gan_not_complete_ssim_2_all_converted_NBI)
    dict_gan_temp_not_complete_ssim = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_temp_not_complete_ssim'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selection_temp_not_complete_ssim = select_specific_cases(df, dict_gan_temp_not_complete_ssim)
    dict_gan_not_complete_ssim = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_not_complete_ssim'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selection_gan_not_complet_ssim = select_specific_cases(df, dict_gan_not_complete_ssim)
    dict_gan_general = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_general'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selection_gan_general = select_specific_cases(df, dict_gan_general)
    dict_gan_general_ssim = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                                               'batch_size': chosen_batch_sizes,
                                               'dataset': ['bladder_tissue_classification_v3_gan_general_ssim'],
                                               'training_data_used': chosen_trained_data,
                                               }
    selecton_an_general_ssim = select_specific_cases(df, dict_gan_general_ssim)

    selection = pd.concat([selection_resnet, selection_not_complete_all_converted_NBI,
                           selection_temp_not_complete_ssim_all_converted_NBI,
                           selection_temp_not_complete_ssim_2_all_converted_NBI,
                           selection_temp_not_complete_ssim,
                           selection_gan_not_complet_ssim,
                           selection_gan_general,
                           selecton_an_general_ssim])

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    x_axis = 'dataset'
    y_axis = metrics_box
    title_plot = 'Comparison base models'  # name_model.replace('_', ' ')
    # daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')


def compare_models_boxplots(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'Matthews CC'
    df = pd.read_csv(dir_to_csv)
    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    chosen_dataset = ['bladder_tissue_classification_v3']
                      #'bladder_tissue_classification_v3_augmented',
                      #'bladder_tissue_classification_v3_gan_new',]

    chosen_trained_data = ['WLI', 'ALL']
    dates_selection = ['21-06-2022', '22-06-2022']

    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,]
                            #'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi',
                            #'temp_not_complete_wli2nbi_ssim']

    dictionary_selection_resnet = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                            }
    selection_resnet = select_specific_cases(df, dictionary_selection_resnet)

    dictionary_selection_densenet = {'name_model': ['densenet121'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   }
    selection_densenet = select_specific_cases(df, dictionary_selection_densenet)

    dictionary_selection_resnet50 = {'name_model': ['resnet50'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    selection_resnet50 = select_specific_cases(df, dictionary_selection_resnet50)

    dictionary_selection_prop = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': ['not_complete_wli2nbi'], 'training_data_used': chosen_trained_data,
                            'date':['21-06-2022', '22-06-2022'],#['06-07-2022', '05-07-2022', '08-07-2022'], # '21-06-2022', '22-06-2022']
                                 }
    selection_proposed = select_specific_cases(df, dictionary_selection_prop)

    temp_not_complete_ssim_selection_dict = {'name_model': ['gan_model_separate_features'],
                                             'learning_rate': chosen_learning_rates,
                                             'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                             'backbone GAN': ['not_complete_wli2nbi_ssim_2'],
                                             'training_data_used': chosen_trained_data,
                                             }
    temp_not_complete_ssim_selection = select_specific_cases(df, temp_not_complete_ssim_selection_dict)
    # here we need to rename 'name_model' so it will appear in the boxplots
    temp_not_complete_ssim_selection.loc[temp_not_complete_ssim_selection['name_model'] == 'gan_model_separate_features', 'name_model'] = 'ssim2'

    dictionary_selection_joint1 = {'name_model': ['gan_model_multi_joint_features'],
                                             'learning_rate': chosen_learning_rates,
                                             'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                             'backbone GAN': ['not_complete_wli2nbi'],
                                             'training_data_used': chosen_trained_data,
                                   }
    selection_proposed_joint1 = select_specific_cases(df, dictionary_selection_joint1)
    dictionary_selection_joint2 = {'name_model': ['gan_model_joint_features_and_domain'],
                                             'learning_rate': chosen_learning_rates,
                                             'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                             'backbone GAN': ['not_complete_wli2nbi'],
                                             'training_data_used': chosen_trained_data,
                                   }
    selection_proposed_joint2 = select_specific_cases(df, dictionary_selection_joint2)
    dictionary_simple_backbone_model = {'name_model': ['simple_model_with_backbones'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'training_data_used': chosen_trained_data,
                                       }
    selection_simple_backbone_model = select_specific_cases(df, dictionary_simple_backbone_model)

    dictionary_simple_separation_model = {'name_model': ['simple_separation_model'],
                                        'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                        'training_data_used': chosen_trained_data,
                                        }
    selection_simple_separation_model = select_specific_cases(df, dictionary_simple_separation_model)

    dictionary_selection_separat_v3 = {'name_model': ['gan_model_separate_features_v3'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    selection_separat_v3 = select_specific_cases(df, dictionary_selection_separat_v3)

    dictionary_selection_separat_v4 = {'name_model': ['gan_separate_features_and_domain_v4'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    selection_separat_v4 = select_specific_cases(df, dictionary_selection_separat_v4)
    dictionary_selection_separat_v5 = {'name_model': ['gan_separate_features_and_domain_v5'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    selection_separat_v5 = select_specific_cases(df, dictionary_selection_separat_v5)

    dict_only_gan_separate_features = {'name_model': ['only_gan_separate_features'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    selection_only_gan_separate_features = select_specific_cases(df, dict_only_gan_separate_features)

    dict_only_gan_model_and_domain = {'name_model': ['only_gan_model_joint_features_and_domain'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    select_only_gan_and_domain = select_specific_cases(df, dict_only_gan_model_and_domain)
    dict_simple_separation_gan_v3 = {'name_model': ['simple_separation_gan_v3'],
                                       'learning_rate': chosen_learning_rates,
                                       'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                       'backbone GAN': ['not_complete_wli2nbi'],
                                       'training_data_used': chosen_trained_data,
                                       }
    select_simple_separation_gan_v3 = select_specific_cases(df, dict_simple_separation_gan_v3)

    dict_simple_semi_sup_1 = {'name_model': ['semi_supervised_resnet101'],
                                     'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v3_semi_sup'],
                                     'training_data_used': chosen_trained_data,
                              'teacher model': ['model_resnet101_fit_lr_1e-05_bs_32_trained_with_WLI_26_07_2022_16_26']
                                     }
    select_simple_semi_sup_1 = select_specific_cases(df, dict_simple_semi_sup_1)

    dict_simple_semi_sup_2= {'name_model': ['semi_supervised_resnet101'],
                                     'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': ['bladder_tissue_classification_v3_semi_sup'],
                                     'training_data_used': chosen_trained_data,
                              'teacher model': ['model_resnet101_fit_lr_1e-05_bs_32_trained_with_WLI_20_06_2022_21_50']
                                     }
    select_simple_semi_sup_2 = select_specific_cases(df, dict_simple_semi_sup_2)
    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    daa.compute_Mann_Whitney_U_test(selection_resnet50, selection_proposed, metric_list=metrics_box)
    daa.compute_Mann_Whitney_U_test(selection_resnet, selection_proposed, metric_list=metrics_box)


    selection_densenet.loc[selection_densenet["name_model"] == "densenet121", "name_model"] = 'ASDASD'
    selection_resnet50.loc[selection_resnet50["name_model"] == "resnet50", "name_model"] = 'A'
    selection_resnet.loc[selection_resnet["name_model"] == "resnet101", "name_model"] = 'B'
    selection_simple_backbone_model.loc[selection_simple_backbone_model["name_model"] == "simple_model_with_backbones", "name_model"] = 'C'
    selection_simple_separation_model.loc[selection_simple_separation_model["name_model"] == "simple_separation_model", "name_model"] = 'D'
    #selection_separat_v4.loc[selection_separat_v4["name_model"] == "gan_separate_features_and_domain_v4", "name_model"] = 'F'
    #selection_separat_v5.loc[selection_separat_v5["name_model"] == "gan_separate_features_and_domain_v5", "name_model"] = 'G'
    selection_only_gan_separate_features.loc[selection_only_gan_separate_features["name_model"] == "only_gan_separate_features", "name_model"] = 'E'
    select_only_gan_and_domain.loc[select_only_gan_and_domain["name_model"] == "only_gan_model_joint_features_and_domain", "name_model"] = 'F'
    select_simple_separation_gan_v3.loc[select_simple_separation_gan_v3["name_model"] == "simple_separation_gan_v3", "name_model"] = 'G'
    selection_separat_v3.loc[selection_separat_v3["name_model"] == "gan_model_separate_features_v3", "name_model"] = 'H'

    selection_proposed.loc[selection_proposed["name_model"] == "gan_model_separate_features", "name_model"] = 'I'
    select_simple_semi_sup_1.loc[select_simple_semi_sup_1["name_model"] == "semi_supervised_resnet101", "name_model"] = 'SS1'
    select_simple_semi_sup_2.loc[select_simple_semi_sup_2["name_model"] == "semi_supervised_resnet101", "name_model"] = 'SS2'


    selection = pd.concat([
                           selection_resnet50,
                           selection_resnet,
        selection_simple_backbone_model,
        selection_simple_separation_model,
        selection_only_gan_separate_features,

        select_only_gan_and_domain,
                           select_simple_separation_gan_v3,
        selection_separat_v3,
        selection_proposed,
        select_simple_semi_sup_2,
        select_simple_semi_sup_1
    ])




    x_axis = 'name_model'
    y_axis = metrics_box
    title_plot = 'Comparison base models'  # name_model.replace('_', ' ')
    # daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')


def compare_base_models(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'Accuracy'
    df = pd.read_csv(dir_to_csv)
    # chosen criteria
    chosen_learning_rates = [0.00001]
    chosen_batch_sizes = [32]
    chosen_dataset = ['bladder_tissue_classification_v3']
                      #'bladder_tissue_classification_v3_augmented',
                      #'bladder_tissue_classification_v3_gan_new',]

    chosen_trained_data = ['WLI', 'ALL']
    dates_selection = ['21-06-2022', '22-06-2022']

    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,]
                            #'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi',
                            #'temp_not_complete_wli2nbi_ssim']

    dictionary_selection_resnet = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                            'date': ['20-06-2022']}
    selection_resnet = select_specific_cases(df, dictionary_selection_resnet)

    dictionary_selection_densenet = {'name_model': ['densenet121'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   }
    selection_densenet = select_specific_cases(df, dictionary_selection_densenet)

    dictionary_selection_resnet50 = {'name_model': ['resnet50'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    selection_resnet50 = select_specific_cases(df, dictionary_selection_resnet50)

    dictionary_selection_prop = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset, 'backbones': ['resnet101_'],
                            'backbone GAN': ['not_complete_wli2nbi'], 'training_data_used': chosen_trained_data,
                            'date':['21-06-2022', '22-06-2022'],
                            #'date':['06-07-2022', '05-07-2022', '08-07-2022'], # '21-06-2022', '22-06-2022']
                                 }
    selection_proposed = select_specific_cases(df, dictionary_selection_prop)

    dictionary_selection_vgg19 = {'name_model': ['vgg19'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    selection_vgg19 = select_specific_cases(df, dictionary_selection_vgg19)

    dictionary_selection_vgg16 = {'name_model': ['vgg16'], 'learning_rate': chosen_learning_rates,
                                  'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                  'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                  }
    selection_vgg16 = select_specific_cases(df, dictionary_selection_vgg16)

    dictionary_selection_inceptionv3 = {'name_model': ['inception_v3'], 'learning_rate': chosen_learning_rates,
                                  'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                  'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                  }
    selection_inceptionv3 = select_specific_cases(df, dictionary_selection_inceptionv3)

    dictionary_selection_xception = {'name_model': ['xception'], 'learning_rate': chosen_learning_rates,
                                        'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                        'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                        }
    selection_xception = select_specific_cases(df, dictionary_selection_xception)

    dictionary_selection_mobilenet = {'name_model': ['mobilenet'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    selection_mobilenet = select_specific_cases(df, dictionary_selection_mobilenet)
    test_selection_dict = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset, 'backbones': ['resnet50_'],
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    test_selection = select_specific_cases(df, test_selection_dict)

    dict_test_selection_dict_2 = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                                     'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset, 'backbones': ['mobilenet_'],
                                     'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                     }
    test_selection_2 = select_specific_cases(df, dict_test_selection_dict_2)

    selection_vgg19.loc[selection_vgg19["name_model"] == "vgg19", "name_model"] = 'A'
    selection_vgg16.loc[selection_vgg16["name_model"] == "vgg16", "name_model"] = 'B'
    selection_inceptionv3.loc[selection_inceptionv3["name_model"] == "inception_v3", "name_model"] = 'C'
    selection_xception.loc[selection_xception["name_model"] == "xception", "name_model"] = 'DX'
    selection_mobilenet.loc[selection_mobilenet["name_model"] == "mobilenet", "name_model"] = 'D'
    selection_densenet.loc[selection_densenet["name_model"] == "densenet121", "name_model"] = 'E'
    selection_resnet.loc[selection_resnet["name_model"] == "resnet101", "name_model"] = 'F'
    selection_resnet50.loc[selection_resnet50["name_model"] == "resnet50", "name_model"] = 'G'
    selection_proposed.loc[selection_proposed["name_model"] == "gan_model_separate_features", "name_model"] = 'H'

    test_selection_2.loc[test_selection_2["name_model"] == "gan_model_separate_features", "name_model"] = 'K'


    test_selection.loc[test_selection["name_model"] == "gan_model_separate_features", "name_model"] = 'R50+'

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

    daa.compute_Mann_Whitney_U_test(selection_resnet, selection_proposed, metric_list=metrics_box)
    daa.compute_Mann_Whitney_U_test(selection_resnet50, test_selection, metric_list=metrics_box)
    daa.compute_Mann_Whitney_U_test(selection_resnet, test_selection, metric_list=metrics_box)

    selection = pd.concat([selection_vgg19,
                           selection_vgg16,
                           selection_inceptionv3,
                           selection_mobilenet,
                           selection_densenet,
                           selection_resnet50,
                           selection_resnet,
                           selection_proposed,
                           ])



    x_axis = 'name_model'
    y_axis = metrics_box
    title_plot = 'Comparison base models'  # name_model.replace('_', ' ')
    # daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
    daa.boxplot_seaborn(selection, x_axis, y_axis, title_plot=title_plot, hue='training_data_used')


def compute_metrics_boxplots(dir_to_csv=(os.path.join(os.getcwd(), 'results', 'sorted_experiments_information.csv')),
                             metrics='all', exclusion_criteria=None):

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
                        'simple_model_with_backbones': 'smwb',
                        'only_gan_separate_features': 'ogsf',
                        'only_gan_model_joint_features_and_domain': 'ogjfd',
                        'gan_separate_features_and_domain_v1': 'gsfadv1',
                        'gan_separate_features_and_domain_v2': 'gsfadv2',
                        'gan_separate_features_and_domain_v3': 'gsfadv3',
                        'gan_separate_features_and_domain_v5': 'gsfadv5',
                        'gan_separate_features_and_domain_v4': 'gsfadv4'}

    selected_models_a = ['gan_model_separate_features']
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

    chosen_trained_data = ['WLI', 'ALL']
    dates_selection = ['21-06-2022', '22-06-2022']

    chosen_gan_backbones = ['not_complete_wli2nbi', '', '', 'NaN', np.nan,]
                            #'not_complete_wli2nbi', 'checkpoint_charlie', 'general_wli2nbi',
                            #'temp_not_complete_wli2nbi_ssim']

    dictionary_selection_resnet = {'name_model': ['resnet101'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                            }
    selection_resnet = select_specific_cases(df, dictionary_selection_resnet)

    dictionary_selection_densenet = {'name_model': ['densenet121'], 'learning_rate': chosen_learning_rates,
                                   'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                                   'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                                   }
    selection_densenet = select_specific_cases(df, dictionary_selection_densenet)

    dictionary_selection_prop = {'name_model': ['gan_model_separate_features'], 'learning_rate': chosen_learning_rates,
                            'batch_size': chosen_batch_sizes, 'dataset': chosen_dataset,
                            'backbone GAN': chosen_gan_backbones, 'training_data_used': chosen_trained_data,
                            'date': dates_selection}
    selection_proposed = select_specific_cases(df, dictionary_selection_prop)

    selection = pd.concat([selection_densenet, selection_resnet, selection_proposed])
    print(selection.keys().tolist())
    print('SELECTION')
    print(selection)
    # 'Accuracy', 'Precision', 'Recall', 'F-1' 'Matthews CC'
    metric_analysis = 'Matthews CC'
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
    x_axis = 'name_model'
    y_axis = metrics_box
    title_plot = 'Comparison base models'#name_model.replace('_', ' ')
    #daa.calculate_p_values(selection, x_axis, y_axis, selected_models)
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
    keys_df = data_frame.keys().tolist()
    new_dict = {k: dictionary_selection[k] for k in dictionary_selection if k in keys_df}
    result_df = data_frame.copy()
    if 'date' in new_dict:
        date_values = new_dict['date']
        result_df = pd.DataFrame()
        for dates in date_values:
            temp_df = data_frame[data_frame['date'].str.contains(dates)]
            result_df = pd.concat([temp_df, result_df])
        del new_dict['date']
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


def analyse_img_quality_experiment(df_real_vals, df_results, plot_rock=False):

    real_vals = list()
    real_conv2nbi = list()
    real_conv2wli = list()

    acc_list = list()
    prec_list = list()
    rec_list = list()
    auc_list = list()

    acc_list_2nbi = list()
    prec_list_2nbi = list()
    rec_list_2nbi = list()
    auc_list_2nbi = list()

    acc_list_2wli = list()
    prec_list_2wli = list()
    rec_list_2wli = list()
    auc_list_2wli = list()

    question_num = df_real_vals['Question'].tolist()
    type_class = df_real_vals['REAL CLASS'].tolist()
    conversion_type = df_real_vals['conversion type'].tolist()
    keys_of_interest = list(range(1, 20))

    for j, num in enumerate(question_num):
        if not np.isnan(num):
            if int(num) in keys_of_interest:
                real_vals.append(int(type_class[j]))
                if conversion_type[j] == 'NBI2WLI':
                    real_conv2wli.append(int(type_class[j]) - 1)
                elif conversion_type[j] == 'WLI2NBI':
                    real_conv2nbi.append(int(type_class[j]) - 1)

    real_vals = [f - 1 for f in real_vals]
    keys_of_interest = ['Q'+str(j)+'.' for j in keys_of_interest]

    for j in range(len(df_results['Timestamp'].tolist())):
        list_results = list()
        list_conv2nbi_results = list()
        list_conv2wli_results = list()
        case_dict = df_results.iloc[j].to_dict()
        new_dict = {k.replace(' ', ''):v for k,v in case_dict.items()}
        new_dict = {k.replace('Pleaseselecttheimagethatlooksmorerealistic', ''):v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythattheimagescorrespondto.', ''): v for k, v in new_dict.items()}
        for key in keys_of_interest:
            question_num = key.replace('Q', '')
            question_num = int(question_num.replace('.', ''))
            list_results.append(int(new_dict[key].replace('Option ', '')))
            selected_case = df_real_vals.iloc[question_num]
            if selected_case['conversion type'] == 'WLI2NBI':
                list_conv2nbi_results.append(int(new_dict[key].replace('Option ', '')) -1)
            elif selected_case ['conversion type'] == 'NBI2WLI':
                list_conv2wli_results.append(int(new_dict[key].replace('Option ', '')) - 1)

        list_results = [f - 1 for f in list_results]

        acc = accuracy_score(real_vals, list_results)
        prec = precision_score(real_vals, list_results)
        rec = recall_score(real_vals, list_results)

        acc_2nbi = accuracy_score(real_conv2nbi, list_conv2nbi_results)
        prec_2nbi = precision_score(real_conv2nbi, list_conv2nbi_results)
        rec_2nbi = recall_score(real_conv2nbi, list_conv2nbi_results)

        acc_2wli = accuracy_score(real_conv2wli, list_conv2wli_results)
        prec_2wli = precision_score(real_conv2wli, list_conv2wli_results)
        rec_2wli = recall_score(real_conv2wli, list_conv2wli_results)

        fpr, tpr, _ = roc_curve(real_vals, list_results)
        auc = roc_auc_score(real_vals, list_results)

        fpr2nbi, tpr2nbi, _ = roc_curve(real_conv2nbi, list_conv2nbi_results)
        auc2nbi = roc_auc_score(real_conv2nbi, list_conv2nbi_results)

        fpr2wli, tpr2wli, _ = roc_curve(real_conv2wli, list_conv2wli_results)
        auc2wli = roc_auc_score(real_conv2wli, list_conv2wli_results)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        auc_list.append(auc)

        acc_list_2nbi.append(acc_2nbi)
        prec_list_2nbi.append(prec_2nbi)
        rec_list_2nbi.append(rec_2nbi)
        auc_list_2nbi.append(auc2nbi)

        acc_list_2wli.append(acc_2wli)
        prec_list_2wli.append(prec_2wli)
        rec_list_2wli.append(rec_2wli)
        auc_list_2wli.append(auc2wli)

        print(f' ALL acc: {acc}, prec: {prec}, rec:{rec}, AUC: {auc}')
        print(f' 2NBI acc: {acc_2nbi}, prec: {prec_2nbi}, rec:{rec_2nbi}, AUC: {auc2nbi}')
        print(f' 2WLI acc: {acc_2wli}, prec: {prec_2wli}, rec:{rec_2wli}, AUC: {auc2wli}')

        if plot_rock is True:
            plt.figure()
            plt.plot([0, 1], [0, 1], color="navy",  linestyle="--")
            plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % auc,)
            plt.plot(fpr2wli, tpr2wli,  label="ROC curve (area = %0.2f)" % auc2wli, )
            plt.plot(fpr2wli, tpr2nbi,  label="ROC curve (area = %0.2f)" % auc2nbi, )
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc='best')
            plt.show()

    print(f'ALL Median Acc: {np.median(acc_list)} +- {np.std(acc_list)}')
    print(f'ALL Median Prec: {np.median(prec_list)} +- {np.std(prec_list)}')
    print(f'ALL Median Rec: {np.median(rec_list)} +- {np.std(rec_list)}')
    print(f'ALL Median AUC: {np.median(auc_list)} +- {np.std(auc_list)}')

    print(f'2NBI Median Acc: {np.median(acc_list_2nbi)} +- {np.std(acc_list_2nbi)}')
    print(f'2NBI Median Prec: {np.median(prec_list_2nbi)} +- {np.std(prec_list_2nbi)}')
    print(f'2NBI Median Rec: {np.median(rec_list_2nbi)} +- {np.std(rec_list_2nbi)}')
    print(f'2NBI Median AUC: {np.median(auc_list_2nbi)} +- {np.std(auc_list_2nbi)}')

    print(f'2WLI Median Acc: {np.median(acc_list_2wli)} +- {np.std(acc_list_2wli)}')
    print(f'2WLI Median Prec: {np.median(prec_list_2wli)} +- {np.std(prec_list_2wli)}')
    print(f'2WLI Median Rec: {np.median(rec_list_2wli)} +- {np.std(rec_list_2wli)}')
    print(f'2WLI Median AUC: {np.median(auc_list_2wli)} +- {np.std(auc_list_2wli)}')


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

    daa.boxplot_users_experiment(dictionary_results)


def analyse_urs_experiments(base_path_dir=None):
    if not base_path_dir:
        base_path_dir = os.path.join(os.getcwd(), 'results')
        path_dir_files = os.path.join(base_path_dir, 'usrs_experiments')

    path_dir_files = os.path.normpath(path_dir_files)
    list_files = [f for f in os.listdir(path_dir_files) if os.path.isfile(os.path.join(path_dir_files, f))]
    result_file = [f for f in list_files if 'results' in f].pop()
    real_vals_file = [f for f in list_files if 'real_values' in f].pop()

    dir_results_file = os.path.join(path_dir_files, result_file)
    dir_vals_file = os.path.join(path_dir_files, real_vals_file)

    df_real_vals = pd.read_csv(dir_vals_file)
    df_results = pd.read_csv(dir_results_file)
    analyse_img_quality_experiment(df_real_vals, df_results)
    specialist_results = select_specific_cases(df_results, {'Role': ['Specialist']})
    print('Specialist results:')
    analyse_img_quality_experiment(df_real_vals, specialist_results)
    resident_results = select_specific_cases(df_results, {'Role': ['Resident']})
    print('Resident results:')
    analyse_img_quality_experiment(df_real_vals, resident_results)

    analyze_diagnosis_experiment(df_real_vals, df_results)


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
        elif type_of_analysis == 'compute_metrics_boxplots':
            compute_metrics_boxplots()
        elif type_of_analysis == 'compare_models_boxplots':
            compare_models_boxplots()
        elif type_of_analysis == 'compare_gans_boxplots':
            compare_gans_boxplots()
        elif type_of_analysis == 'compare_dataset_boxplots':
            compare_dataset_boxplots()
        elif type_of_analysis == 'compare_base_models':
            compare_base_models()
        elif type_of_analysis == 'compare_base_datasets':
            compare_datasets_v1()
        elif type_of_analysis == 'fid_score_vs_metrics':
            fid_score_vs_metrics()
        elif type_of_analysis == 'prepare_data':
            prepare_data(directory_to_analyze)
        elif type_of_analysis == 'analyse_urs_experiments':
            analyse_urs_experiments()


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
