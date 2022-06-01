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
        daa.save_yaml(dir_data_yaml, performance_resume)

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


def compute_metrics_boxplots():
    pass


def main(_argv):

    type_of_analysis = FLAGS.type_of_analysis
    directory_to_analyze = FLAGS.directory_to_analyze
    ground_truth_file = FLAGS.ground_truth_file
    plot_figures = FLAGS.plot_figures
    display_figures = FLAGS.display_figures

    if type_of_analysis and directory_to_analyze:
        if type_of_analysis == 'compute_metrics_batch_experiments':
            compute_metrics_batch_experiments(directory_to_analyze, gt_file=ground_truth_file)
        if type_of_analysis == 'compute_metrics_boxplots':
            compute_metrics_boxplots(directory_to_analyze, gt_file=ground_truth_file)


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

    try:
        app.run(main)
    except SystemExit:
        pass
