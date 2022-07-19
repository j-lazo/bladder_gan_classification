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
    selection_wli2nbi.loc[selection_wli2nbi["backbone GAN"] == "wli2nbi", "backbone GAN"] = 'B'
    base_selection.loc[base_selection["backbone GAN"] == "not_complete_wli2nbi", "backbone GAN"] = 'C'
    general_selection.loc[general_selection["backbone GAN"] == "general_wli2nbi", "backbone GAN"] = 'D'
    selection_wli2nbi_ssim.loc[selection_wli2nbi_ssim["backbone GAN"] == "wli2nbi_ssim", "backbone GAN"] = 'E'
    not_complete_ssim_selection_2.loc[not_complete_ssim_selection_2["backbone GAN"] == "not_complete_wli2nbi_ssim_2", "backbone GAN"] = 'F'
    general_ssim_selection.loc[general_ssim_selection["backbone GAN"] == "general_wli2nbi_ssim", "backbone GAN"] = 'G'

    selection = pd.concat([selection_resnet,
                           selection_wli2nbi,
                           base_selection,
                           general_selection,
                           selection_wli2nbi_ssim,
                           not_complete_ssim_selection_2,
                           general_ssim_selection,
                           ])

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

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
                            #'date':['21-06-2022', '22-06-2022'],#['06-07-2022', '05-07-2022', '08-07-2022'], # '21-06-2022', '22-06-2022']
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

    selection_densenet.loc[selection_densenet["name_model"] == "densenet121", "name_model"] = 'A'
    selection_resnet.loc[selection_resnet["name_model"] == "resnet101", "name_model"] = 'B'
    selection_proposed.loc[selection_proposed["name_model"] == "gan_model_separate_features", "name_model"] = 'C'
    selection_simple_backbone_model.loc[selection_simple_backbone_model["name_model"] == "simple_model_with_backbones", "name_model"] = 'D'
    selection_simple_separation_model.loc[selection_simple_separation_model["name_model"] == "simple_separation_model", "name_model"] = 'E'
    selection_separat_v3.loc[selection_separat_v3["name_model"] == "gan_model_separate_features_v3", "name_model"] = 'F'
    selection_separat_v4.loc[selection_separat_v4["name_model"] == "gan_separate_features_and_domain_v4", "name_model"] = 'G'
    selection_separat_v5.loc[selection_separat_v5["name_model"] == "gan_separate_features_and_domain_v5", "name_model"] = 'H'
    selection_only_gan_separate_features.loc[selection_only_gan_separate_features["name_model"] == "only_gan_separate_features", "name_model"] = 'I'
    select_only_gan_and_domain.loc[select_only_gan_and_domain["name_model"] == "only_gan_model_joint_features_and_domain", "name_model"] = 'J'
    select_simple_separation_gan_v3.loc[select_simple_separation_gan_v3["name_model"] == "simple_separation_gan_v3", "name_model"] = 'X'
    selection_resnet50.loc[selection_resnet50["name_model"] == "resnet50", "name_model"] = '50'

    selection = pd.concat([selection_densenet,
                           selection_resnet50,
                           selection_resnet,
                           selection_proposed,
                           select_simple_separation_gan_v3,])
                           #selection_simple_backbone_model,
                           #selection_simple_separation_model,
                           #selection_separat_v3,
                           #selection_separat_v4,
                           #selection_separat_v5,
                           #selection_only_gan_separate_features,
                           #select_only_gan_and_domain,
                           #])


    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

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
    print('TEST 2', test_selection_2)

    selection_densenet.loc[selection_densenet["name_model"] == "densenet121", "name_model"] = 'D121'
    selection_resnet.loc[selection_resnet["name_model"] == "resnet101", "name_model"] = 'R101'
    selection_resnet50.loc[selection_resnet50["name_model"] == "resnet50", "name_model"] = 'R50'
    selection_vgg19.loc[selection_vgg19["name_model"] == "vgg19", "name_model"] = 'VG19'
    selection_inceptionv3.loc[selection_inceptionv3["name_model"] == "inception_v3", "name_model"] = 'IV3'
    selection_vgg16.loc[selection_vgg16["name_model"] == "vgg16", "name_model"] = 'VG16'
    selection_xception.loc[selection_xception["name_model"] == "xception", "name_model"] = 'XC'
    selection_mobilenet.loc[selection_mobilenet["name_model"] == "mobilenet", "name_model"] = 'MN'
    test_selection.loc[test_selection["name_model"] == "gan_model_separate_features", "name_model"] = 'R50+'
    selection_proposed.loc[selection_proposed["name_model"] == "gan_model_separate_features", "name_model"] = 'R101+'
    test_selection_2.loc[test_selection_2["name_model"] == "gan_model_separate_features", "name_model"] = 'MN+'


    selection = pd.concat([selection_vgg19,
                           selection_vgg16,
                           selection_inceptionv3,
                           selection_mobilenet,
                           selection_densenet,
                           selection_resnet50,
                           selection_resnet,
                           test_selection,
                           selection_proposed,
                           ])

    metrics_box = ['ALL', 'WLI', 'NBI']
    metrics_box = [''.join([metric_analysis, ' ', m]) for m in metrics_box]

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
    metric_analysis = 'Accuracy'
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

    print('RESULT')
    print(result_df)
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
    acc_list = list()
    prec_list = list()
    rec_list = list()
    auc_list = list()
    question_num = df_real_vals['Question'].tolist()
    type_class = df_real_vals['REAL CLASS'].tolist()
    keys_of_interest = list(range(1, 20))

    for j, num in enumerate(question_num):
        if not np.isnan(num):
            if int(num) in keys_of_interest:
                real_vals.append(int(type_class[j]))

    real_vals = [f - 1 for f in real_vals]
    keys_of_interest = ['Q'+str(j)+'.' for j in keys_of_interest]

    for j in range(len(df_results['Timestamp'].tolist())):
        list_results = list()
        case_dict = df_results.iloc[j].to_dict()
        new_dict = {k.replace(' ', ''):v for k,v in case_dict.items()}
        new_dict = {k.replace('Pleaseselecttheimagethatlooksmorerealistic', ''):v for k, v in new_dict.items()}
        new_dict = {k.replace('Pleasechoosethecategorythattheimagescorrespondto.', ''): v for k, v in new_dict.items()}
        for key in keys_of_interest:
            list_results.append(int(new_dict[key].replace('Option ', '')))

        list_results = [f - 1 for f in list_results]
        acc = accuracy_score(real_vals, list_results)
        prec = precision_score(real_vals, list_results)
        rec = recall_score(real_vals, list_results)

        fpr, tpr, _ = roc_curve(real_vals, list_results)
        auc = roc_auc_score(real_vals, list_results)

        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        auc_list.append(auc)

        print(f'acc: {acc}, prec: {prec}, rec:{rec}, AUC: {auc}')
        if plot_rock is True:
            plt.figure()
            plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %0.2f)" % auc,)
            plt.plot([0, 1], [0, 1], color="navy",  linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver operating characteristic example")
            plt.legend(loc='best')
            plt.show()

    print(f'Median Acc: {np.median(acc_list)} +- {np.std(acc_list)}')
    print(f'Median Prec: {np.median(prec_list)} +- {np.std(prec_list)}')
    print(f'Median Rec: {np.median(rec_list)} +- {np.std(rec_list)}')
    print(f'Median AUC: {np.median(auc_list)} +- {np.std(auc_list)}')


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
