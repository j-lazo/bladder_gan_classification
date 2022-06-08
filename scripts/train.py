from absl import app, flags
from absl.flags import FLAGS
from utils.data_management import datasets as dam
from utils.data_management import file_management as fam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam
import datetime
from models.model_utils import *
from models.gan_classification import *
from models.simple_models import *
import shutil
import random

def compile_model(name_model, strategy, optimizer, loss, metrics,
                  backbones=None, gan_weights=None):

    if strategy:
        with strategy.scope():

            if name_model == 'gan_model_multi_joint_features':
                model = build_gan_model_joint_features(backbones=backbones, gan_weights=gan_weights)
                model.summary()
                print('Multi-GPU training')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            elif name_model == 'gan_model_separate_features':
                model = build_gan_model_separate_features(backbones=backbones, gan_weights=gan_weights)
                model.summary()
                print('Multi-GPU training')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            else:
                model = build_pretrained_model(name_model)

    else:
        if name_model == 'gan_model_multi_joint_features':
            model = build_gan_model_joint_features(backbones=backbones, gan_weights=gan_weights)
        elif name_model == 'gan_model_separate_features':
            model = build_gan_model_separate_features(backbones=backbones, gan_weights=gan_weights)
        else:
            model = build_pretrained_model(name_model)

        model.summary()
        # compile model
        print('Single-GPU training')
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def call_models(name_model, path_dataset, mode='fit', backbones=['resnet101'], gan_model='checkpoint_charlie', epochs=2,
                batch_size=1, learning_rate=0.001, val_data=None, test_data=None, eval_val_set=False, eval_train_set=False,
                results_dir=os.path.join(os.getcwd(), 'results'), gpus_available=None, analyze_data=False,
                specific_domain=None, prepare_finished_experiment=False, k_folds={None}, val_division=0.2):

    multi_input_models = ['gan_model_multi_joint_features', 'gan_model_separate_features',
                          'gan_model_joint_features_and_domain', 'simple_model_domain_input']

    if name_model not in multi_input_models:
        multioutput = False
    else:
        multioutput = True

    gan_pretrained_weights = os.path.join(os.getcwd(), 'scripts', 'models', 'weights_gans', gan_model)

    # create the file_dir to save the results

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                                batch_size=batch_size, backbone_model=backbone_model,
                                                mode=mode, specific_domain=specific_domain)

    results_directory = ''.join([results_dir, '/', new_results_id, '/'])
    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    for fold in k_folds:
        print('fold:', fold)
        # prepare the data
        list_subdirs_dataset = os.listdir(path_dataset)
        if fold:
            csv_file = [f for f in os.listdir(path_dataset) if f.endswith('.csv')].pop()
            path_csv_file = os.path.join(path_dataset, csv_file)
            list_train_cases = list()
            for n_fold in k_folds:
                if n_fold != fold:
                    list_train_cases += [''.join(['case_', str(num_case).zfill(3)]) for num_case in k_folds[n_fold]]

            test_data = path_dataset
            list_test_cases = [''.join(['case_', str(num_case).zfill(3)]) for num_case in k_folds[fold]]
            x_input, dictionary_files = dam.load_data_from_directory(path_dataset,
                                                                     csv_annotations=path_csv_file,
                                                                     specific_domain=specific_domain,
                                                                     case_specific=list_train_cases)
            # option 1 randomly divide the dataset into train/val
            random.shuffle(x_input)
            val_x = [f for f in x_input if random.random() <= val_division]
            train_x = [f for f in x_input if f not in val_x]
            dic_train = {x: dictionary_files[x] for x in train_x}
            dic_val = {x: dictionary_files[x] for x in val_x}

            # option 2, select case_specific folds as train / val
            # pass
            train_dataset = dam.make_tf_dataset(train_x, dic_train, batch_size,
                                                training=True, multi_output=multioutput,
                                                specific_domain=specific_domain)
            val_dataset = dam.make_tf_dataset(val_x, dic_val, batch_size,
                                              training=True, multi_output=multioutput, specific_domain=specific_domain)
        else:
            if 'train' in list_subdirs_dataset:
                path_train_dataset = os.path.join(path_dataset, 'train')
            else:
                path_train_dataset = path_dataset

            if 'val' in list_subdirs_dataset:
                path_val_dataset = os.path.join(path_dataset, 'val')
            else:
                path_val_dataset = val_data

            csv_file_train = [f for f in os.listdir(path_train_dataset) if f.endswith('.csv')].pop()
            path_csv_file_train = os.path.join(path_train_dataset, csv_file_train)
            train_x, dictionary_train = dam.load_data_from_directory(path_train_dataset, csv_annotations=path_csv_file_train,
                                                                     specific_domain=specific_domain)

            train_dataset = dam.make_tf_dataset(train_x, dictionary_train, batch_size,
                                                training=True, multi_output=multioutput, specific_domain=specific_domain)

            csv_file_val = [f for f in os.listdir(path_val_dataset) if f.endswith('.csv')].pop()
            path_csv_file_val = os.path.join(path_val_dataset, csv_file_val)
            val_x, dictionary_val = dam.load_data_from_directory(path_val_dataset, csv_annotations=path_csv_file_val,
                                                                 specific_domain=specific_domain)
            val_dataset = dam.make_tf_dataset(val_x, dictionary_val, batch_size,
                                              training=True, multi_output=multioutput, specific_domain=specific_domain)

        train_steps = len(train_x) // batch_size
        val_steps = len(val_x) // batch_size

        if len(train_x) % batch_size != 0:
            train_steps += 1
        if len(val_x) % batch_size != 0:
            val_steps += 1

        if len(gpus_available) > 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            strategy = None

        if fold:
            temp_name_model = results_directory + new_results_id + ''.join(['_model_', fold, '.h5'])
        else:
            temp_name_model = results_directory + new_results_id + '_model.h5'
        # load and compile the  model
        #model = compile_model(name_model, strategy, optimizer, loss, metrics, backbones=backbones,
        #                      gan_weights=gan_pretrained_weights)

        if strategy:
            with strategy.scope():
                optimizer = Adam(learning_rate=learning_rate)
                metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                loss = 'categorical_crossentropy'

                if name_model == 'gan_model_multi_joint_features':
                    model = build_gan_model_joint_features(backbones=backbones, gan_weights=gan_pretrained_weights)
                elif name_model == 'gan_model_separate_features':
                    model = build_gan_model_separate_features(backbones=backbones, gan_weights=gan_pretrained_weights)
                elif name_model == 'gan_model_joint_features_and_domain':
                    model = build_gan_model_joint_features_and_domain(backbones=backbones, gan_weights=gan_pretrained_weights)
                elif name_model == 'simple_model_domain_input':
                    model = build_simple_model_with_domain_input(backbones=backbones)   
                else:
                    model = build_pretrained_model(name_model)

                model.summary()
                print('Multi-GPU training')
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        else:
            if name_model == 'gan_model_multi_joint_features':
                model = build_gan_model_joint_features(backbones=backbones, gan_weights=gan_weights)
            elif name_model == 'gan_model_separate_features':
                model = build_gan_model_separate_features(backbones=backbones, gan_weights=gan_weights)
            elif name_model == 'gan_model_joint_features_and_domain':
                model = build_gan_model_joint_features_and_domain(backbones=backbones, gan_weights=gan_pretrained_weights)
            elif name_model == 'simple_model_domain_input':
                model = build_simple_model_with_domain_input(backbones=backbones)
            else:
                model = build_pretrained_model(name_model)

            model.summary()
            # compile model
            print('Single-GPU training')
            optimizer = Adam(learning_rate=learning_rate)
            metrics = ["accuracy", tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')]
            loss = 'categorical_crossentropy'
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if fold:
            history_name = ''.join([results_directory, 'train_history_', fold, '_', new_results_id, "_.csv"])
            dir_save_model = ''.join([results_directory, 'model_', fold, '_', new_results_id])

        else:
            history_name = ''.join([results_directory, 'train_history_', new_results_id, "_.csv"])
            dir_save_model = ''.join([results_directory, 'model_', new_results_id])

        if mode == 'fit':
            callbacks = [
                ModelCheckpoint(temp_name_model,
                                monitor="val_loss", save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', patience=25),
                CSVLogger(history_name),
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
            start_time = datetime.datetime.now()

            trained_model = model.fit(train_dataset,
                                      epochs=epochs,
                                      shuffle=True,
                                      batch_size=batch_size,
                                      validation_data=val_dataset,
                                      steps_per_epoch=train_steps,
                                      validation_steps=val_steps,
                                      verbose=True,
                                      callbacks=callbacks)

            model.save(dir_save_model)

            print('Total Training TIME:', (datetime.datetime.now() - start_time))
            print('History Model Keys:')
            print(trained_model.history.keys())
            # in case evaluate val dataset is True
            if eval_val_set is True:
                loaded_model, _ = load_model(dir_save_model)
                evaluate_and_predict(loaded_model, val_dataset, results_directory,
                                     results_id=new_results_id, output_name='val')

            if eval_train_set is True:
                loaded_model, _ = load_model(dir_save_model)
                evaluate_and_predict(loaded_model, train_dataset, results_directory,
                                     results_id=new_results_id, output_name='train')

            if 'test' in list_subdirs_dataset:
                loaded_model, _ = load_model(dir_save_model)
                path_test_dataset = os.path.join(path_dataset, 'test')
                print(f'Test directory found at: {path_test_dataset}')
                evalute_test_directory(loaded_model, path_test_dataset, results_directory, new_results_id,
                                       analyze_data=analyze_data, multioutput=multioutput)

            else:
                path_test_dataset = test_data
                loaded_model, _ = load_model(dir_save_model)
                print(f'Test directory found at: {path_test_dataset}')
                evalute_test_directory(loaded_model, path_test_dataset, results_directory, new_results_id,
                                       analyze_data=analyze_data, list_test_cases=list_test_cases, fold=fold,
                                       multioutput=multioutput)

    tf.keras.backend.clear_session()
    if prepare_finished_experiment:
        compressed_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_compressed')
        transfer_results = os.path.join(os.getcwd(), 'results', 'bladder_tissue_classification_v2_transfer')
        path_folder = results_directory
        fam.compress_files(path_folder, destination_dir=compressed_results)
        # make a temporal folder with only the files you want to compress
        experiment_folder = os.path.split(os.path.normpath(path_folder))[-1]
        temporal_folder_dir = os.path.join(os.getcwd(), 'results', 'remove_folders', experiment_folder)
        os.mkdir(temporal_folder_dir)
        # move files to temporal folder
        list_files = [f for f in os.listdir(path_folder) if f.endswith('.png') or
                      f.endswith('.yaml') or f.endswith('.csv')]
        # temporal folder to delete
        for file_name in list_files:
            shutil.copyfile(os.path.join(path_folder, file_name), os.path.join(temporal_folder_dir, file_name))
        # compress the file
        fam.compress_files(temporal_folder_dir, destination_dir=transfer_results)
        print(f'Experiment finished, results compressed and saved in {transfer_results}')


def main(_argv):

    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))
    path_dataset = FLAGS.path_dataset
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    results_dir = FLAGS.results_dir
    learning_rate = FLAGS.learning_rate
    analyze_data = FLAGS.analyze_data
    backbones = FLAGS.backbones
    specific_domain = FLAGS.specific_domain
    prepare_finished_experiment = FLAGS.prepare_finished_experiment
    k_folds = FLAGS.k_folds

    if k_folds:
        folds_dict = {'fold_1': [26, 14, 25, 12, 18, 17, 16, 32],
                      'fold_2': [3, 28, 23, 20, 9, 8, 31],
                      'fold_3': [27, 21, 6, 13, 11, 4, 2, 30],
                      'fold_4': [1, 24, 22, 7, 5, 10, 29]}

        call_models(name_model, path_dataset, batch_size=batch_size, gpus_available=physical_devices,
                epochs=epochs, results_dir=results_dir, learning_rate=learning_rate, analyze_data=analyze_data,
                backbones=backbones, specific_domain=specific_domain,
                prepare_finished_experiment=prepare_finished_experiment, k_folds=folds_dict)

    else:
        call_models(name_model, path_dataset, batch_size=batch_size, gpus_available=physical_devices,
                    epochs=epochs, results_dir=results_dir, learning_rate=learning_rate, analyze_data=analyze_data,
                    backbones=backbones, specific_domain=specific_domain,
                    prepare_finished_experiment=prepare_finished_experiment)


if __name__ == '__main__':

    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('path_dataset', '', 'directory dataset')
    flags.DEFINE_string('val_dataset', '', 'directory val dataset')
    flags.DEFINE_string('test_dataset', '', 'directory test dataset')
    flags.DEFINE_integer('batch_size', 8, 'batch size')
    flags.DEFINE_integer('epochs', 1, 'epochs')
    flags.DEFINE_string('results_dir', os.path.join(os.getcwd(), 'results'), 'directory to save the results')
    flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
    flags.DEFINE_boolean('analyze_data', True, 'analyze the data after the experiment')
    flags.DEFINE_list('backbones', ['resnet101'], 'A list of the nets used as backbones: resnet101, resnet50, densenet121, vgg19')
    flags.DEFINE_string('specific_domain', None, 'In case a specific domain wants to be selected for training and validation data ops:'
                                                 '[NBI, WLI]')
    flags.DEFINE_boolean('prepare_finished_experiment', False, 'either compress the files and move them to the transfer folder or not')
    flags.DEFINE_boolean('k_folds', False, 'in case k-folds cross validation is taking palce')

    try:
        app.run(main)
    except SystemExit:
        pass