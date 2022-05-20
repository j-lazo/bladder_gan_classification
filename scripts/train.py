import os
from absl import app, flags
from absl.flags import FLAGS
from models.gan_classification import *
from utils.data_management import datasets as dam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Nadam


def call_models(name_model, path_dataset, mode='fit', backbones=['resnet101'], gan_model='checkpoint_charlie', epochs=2, batch_size=1,
                learning_rate=0.001, val_data=None, test_data=None, results_dir=os.path.join(os.getcwd(), 'results')):

    gan_pretrained_weights = os.path.join(os.getcwd(), 'scripts', 'models', 'weights_gans', gan_model)
    # prepare the data
    list_subdirs_dataset = os.listdir(path_dataset)
    if 'train' in list_subdirs_dataset:
        path_train_dataset = os.path.join(path_dataset, 'train')
    else:
        path_train_dataset = path_dataset

    if 'val' in list_subdirs_dataset:
        path_val_dataset = os.path.join(path_dataset, 'val')
    else:
        path_val_dataset = val_data

    if 'test' in list_subdirs_dataset:
        path_test_dataset = os.path.join(path_dataset, 'test')
    else:
        path_test_dataset = test_data

    # create the file_dir to save the results

    # ID name for the folder and results
    backbone_model = ''.join([name_model + '_' for name_model in backbones])
    new_results_id = dam.generate_experiment_ID(name_model=name_model, learning_rate=learning_rate,
                                            batch_size=batch_size, backbone_model=backbone_model,
                                            mode=mode)

    results_directory = ''.join([results_dir, new_results_id, '/'])
    temp_name_model = results_directory + new_results_id + "_model.h5"

    # if results experiment doesn't exists create it
    if not os.path.isdir(results_directory):
        os.mkdir(results_directory)

    csv_file_train = [f for f in os.listdir(path_train_dataset) if f.endswith('.csv')].pop()
    path_csv_file_train = os.path.join(path_train_dataset, csv_file_train)
    train_x, dictionary_train = dam.load_data_from_directory(path_train_dataset,
                                                            csv_annotations=path_csv_file_train)

    csv_file_val = [f for f in os.listdir(path_val_dataset) if f.endswith('.csv')].pop()
    path_csv_file_val = os.path.join(path_val_dataset, csv_file_val)
    val_x, dictionary_val = dam.load_data_from_directory(path_val_dataset,
                                                             csv_annotations=path_csv_file_val)

    train_dataset = dam.make_tf_dataset(path_train_dataset, batch_size)
    val_dataset = dam.make_tf_dataset(path_val_dataset, batch_size)

    # load the model
    if name_model == 'build_gan_model_features':
        model = build_gan_model_features(backbones=backbones, gan_weights=gan_pretrained_weights)
        model.summary()

        train_steps = len(train_x) // batch_size
        val_steps = len(val_x) // batch_size

        if len(train_x) % batch_size != 0:
            train_steps += 1
        if len(val_x) % batch_size != 0:
            val_steps += 1

        callbacks = [
            ModelCheckpoint(temp_name_model,
                            monitor="val_loss", save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', patience=25),
            CSVLogger(results_directory + 'train_history_' + new_results_id + "_.csv"),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

    # compile_model
    optimizer = Adam(learning_rate=learning_rate)
    loss = 'categorical_crossentropy'
    metrics = ["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

    if mode == 'fit':
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # train the model
        trained_model = model.fit(train_dataset,
                                  epochs=epochs,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  validation_data=val_dataset,
                                  steps_per_epoch=train_steps,
                                  validation_steps=val_steps,
                                  verbose=True,
                                  callbacks=callbacks)


def main(_argv):
    path_dataset = FLAGS.path_dataset
    name_model = FLAGS.name_model
    batch_size = FLAGS.batch_size
    call_models(name_model, path_dataset, batch_size=batch_size)


if __name__ == '__main__':
    flags.DEFINE_string('name_model', '', 'name of the model')
    flags.DEFINE_string('path_dataset', '', 'name of the model')
    flags.DEFINE_string('val_dataset', '', 'name of the model')
    flags.DEFINE_string('test_dataset', '', 'name of the model')
    flags.DEFINE_integer('batch_size', 8, 'batch size')

    try:
        app.run(main)
    except SystemExit:
        pass