import tensorflow as tf
import os
from tensorflow.keras import applications
import time
import tqdm
from utils.data_management import datasets as dam
import numpy as np
import pandas as pd

class Checkpoint:
    """Enhanced "tf.train.Checkpoint"."""

    def __init__(self,
                 checkpoint_kwargs,  # for "tf.train.Checkpoint"
                 directory,  # for "tf.train.CheckpointManager"
                 max_to_keep=5,
                 keep_checkpoint_every_n_hours=None):
        self.checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep, keep_checkpoint_every_n_hours)

    def restore(self, save_path=None):
        save_path = self.manager.latest_checkpoint if save_path is None else save_path
        return self.checkpoint.restore(save_path)

    def save(self, file_prefix_or_checkpoint_number=None, session=None):
        if isinstance(file_prefix_or_checkpoint_number, str):
            return self.checkpoint.save(file_prefix_or_checkpoint_number, session=session)
        else:
            return self.manager.save(checkpoint_number=file_prefix_or_checkpoint_number)

    def __getattr__(self, attr):
        if hasattr(self.checkpoint, attr):
            return getattr(self.checkpoint, attr)
        elif hasattr(self.manager, attr):
            return getattr(self.manager, attr)
        else:
            self.__getattribute__(attr)  # this will raise an exception


def load_pretrained_backbones(name_model, weights='imagenet', include_top=False, trainable=False, new_name=None):

    """
    Loads a pretrained model given a name
    :param name_model: (str) name of the model
    :param weights: (str) weights names (default imagenet)
    :return: sequential model with the selected weights
    """

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}

    base_dir_weights = ''.join([os.getcwd(), '/scripts/models/weights_pretrained_backbones/'])
    if name_model == 'vgg16':
        weights_dir = base_dir_weights + 'vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg16.VGG16(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'vgg19':
        weights_dir = base_dir_weights + 'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.vgg19.VGG19(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'inception_v3':
        weights_dir = base_dir_weights + 'inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.inception_v3.InceptionV3(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'resnet50':
        weights_dir = base_dir_weights + 'resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet50.ResNet50(include_top=include_top, weights=weights_dir)
        base_model.trainable = True

    elif name_model == 'resnet101':
        weights_dir = base_dir_weights + 'resnet101/resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.resnet.ResNet101(include_top=include_top, weights=weights_dir)
        base_model.trainable = True

    elif name_model == 'mobilenet':
        weights_dir = base_dir_weights + 'mobilenet/mobilenet_1_0_224_tf_no_top.h5'
        base_model = applications.mobilenet.MobileNet(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'densenet121':
        weights_dir = base_dir_weights + 'densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.densenet.DenseNet121(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    elif name_model == 'xception':
        weights_dir = base_dir_weights + 'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_model = applications.xception.Xception(include_top=include_top, weights=weights_dir)
        base_model.trainable = trainable

    else:
        raise ValueError(f' MODEL: {name_model} not found')

    new_base_model = tf.keras.models.clone_model(base_model)
    new_base_model.set_weights(base_model.get_weights())

    return new_base_model


def evalute_test_directory(model, test_data, results_directory, new_results_id, analyze_data=True):

    # determine if there are sub_folders or if it's the absolute path of the dataset
    sub_dirs = [f for f in os.listdir(test_data) if os.path.isdir(test_data + f)]
    if sub_dirs:
        print(f'sub-directoires {sub_dirs} found in test foler')
        for sub_dir in sub_dirs:
            test_data_dir = ''.join([test_data, '/', sub_dir])
            name_file = evaluate_and_predict(model, test_data_dir, results_directory,
                                             results_id=new_results_id, output_name='test',
                                             analyze_data=analyze_data)
            print(f'Evaluation results saved at {name_file}')

    else:
        name_file = evaluate_and_predict(model, test_data, results_directory,
                                         results_id=new_results_id, output_name='test',
                                         analyze_data=analyze_data)

        print(f'Evaluation results saved at {name_file}')


def evaluate_and_predict(model, directory_to_evaluate, results_directory,
                         output_name='', results_id='', batch_size=1,
                         analyze_data=False, output_dir=''):
    print(f'Evaluation of: {directory_to_evaluate}')

    # load the data to evaluate and predict

    batch_size = 8
    test_x, test_y, dataset_dictionary = dam.load_data_from_directory(directory_to_evaluate)
    test_dataset = dam.make_tf_dataset(directory_to_evaluate, batch_size)
    test_steps = (len(test_x) // batch_size)

    if len(test_x) % batch_size != 0:
        test_steps += 1

    evaluation = model.evaluate(test_dataset, steps=test_steps)
    inference_times = []
    prediction_names = []
    prediction_outputs = []
    real_values = []
    print('Evaluation results:')
    print(evaluation)
    for i, (x, y) in tqdm.tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        real_values.append(dataset_dictionary[x])
        prediction_names.append(os.path.split(x)[-1])
        init_time = time.time()
        x = read_stacked_images_npy_predict(x)
        x = np.expand_dims(x, axis=0)
        y_pred = model.predict(x)
        prediction_outputs.append(y_pred[0])
        inference_times.append(time.time() - init_time)
        # determine the top-1 prediction class
        prediction_id = np.argmax(y_pred[0])

    print('Prediction times analysis')
    print(np.min(prediction_outputs), np.mean(prediction_outputs), np.max(prediction_outputs))

    unique_values = np.unique(real_values)
    label_index = [unique_values[np.argmax(pred)] for pred in prediction_outputs]

    x_pred = [[] for _ in range(len(unique_values))]
    for x in prediction_outputs:
        for i in range(len(x)):
            x_pred[i].append(x[i])

    header_column = ['class_' + str(i+1) for i in range(5)]
    header_column.insert(0, 'fname')
    header_column.append('over all')
    header_column.append('real values')
    df = pd.DataFrame(columns=header_column)
    df['fname'] = [os.path.basename(x_name) for x_name in test_x]
    df['real values'] = real_values
    for i in range(len(unique_values)):
        class_name = 'class_' + str(i+1)
        df[class_name] = x_pred[i]

    df['over all'] = label_index
    # save the predictions  of each case
    results_csv_file = ''.join([results_directory, 'predictions_', output_name, '_', results_id, '_.csv'])
    df.to_csv(results_csv_file, index=False)

    if analyze_data is True:
        pass

    return results_csv_file