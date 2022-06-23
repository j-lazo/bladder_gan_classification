import tensorflow as tf
from tensorflow import keras
from models.building_blocks import *
from models.model_utils import *

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299),
                      'simple_residual_model': (256, 256)}


def simple_residual_model(input_size=256, num_filters=[32, 64, 128, 256]):
    input_layer = Input((input_size, input_size, 3))
    x = input_layer
    for f in num_filters[:-1]:
        x = residual_block(x, f)
        x = MaxPool2D((2, 2))(x)

    output_layer = residual_block(x, num_filters[-1])

    return Model(inputs=input_layer, outputs=output_layer, name='simple_residual_model')


def build_simple_model_with_domain_input(num_classes, backbones):
    # inputs
    name_model = backbones[0]
    input_image = keras.Input(shape=(256, 256, 3), name="image")
    t_input = keras.Input(shape=(1,), name="img_domain")

    x = tf.image.resize(input_image, input_sizes_models[name_model], method='bilinear')
    x = get_preprocess_input_backbone(name_model, x)
    base_model = load_pretrained_backbones(name_model)
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Concatenate()([x, t_input])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='simple_model_with_domain')


def build_simple_separation_model(num_classes, backbones):
    # inputs
    name_model = backbones[0]
    input_image = keras.Input(shape=(256, 256, 3), name="image")
    t_input = keras.Input(shape=(1,), name="img_domain")

    x1 = tf.image.resize(input_image, input_sizes_models[name_model], method='bilinear')
    x1 = get_preprocess_input_backbone(name_model, x1)
    base_model = load_pretrained_backbones(name_model)
    for layer in base_model.layers:
        layer.trainable = False

    x1 = base_model(x1)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(num_classes, activation='softmax')(x1)

    x2 = tf.image.resize(input_image, input_sizes_models['simple_residual_model'], method='bilinear')/255.
    base_model_2 = simple_residual_model()
    for layer in base_model_2.layers:
        layer.trainable = True

    x2 = base_model_2(x2)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(num_classes, activation='softmax')(x2)

    output_layer = tf.keras.backend.switch(t_input, x1, x2)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='simple_model_with_domain')


def build_simple_separation_with_backbones(num_classes, backbones):
    if len(backbones) < 2:
        backbones = backbones * 2
    # inputs
    name_model = backbones[0]
    input_image = keras.Input(shape=(256, 256, 3), name="image")
    t_input = keras.Input(shape=(1,), name="img_domain")

    x1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')
    x1 = get_preprocess_input_backbone(backbones[0], x1)
    base_model = load_pretrained_backbones(backbones[0])
    for layer in base_model.layers:
        layer.trainable = False
    base_model._name = 'backbone_1'

    x1 = base_model(x1)
    x1 = GlobalAveragePooling2D()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)
    x1 = Dense(512, activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(num_classes, activation='softmax')(x1)

    x2 = tf.image.resize(input_image, input_sizes_models[backbones[1]], method='bilinear')
    x2 = get_preprocess_input_backbone(backbones[0], x2)
    base_model_2 = load_pretrained_backbones(backbones[1])
    for layer in base_model_2.layers:
        layer.trainable = False
    base_model_2._name = 'backbone_2'

    x2 = base_model_2(x2)
    x2 = GlobalAveragePooling2D()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    x2 = Dense(512, activation='relu')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(num_classes, activation='softmax')(x2)

    output_layer = tf.keras.backend.switch(t_input, x1, x2)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='simple_model_with_backbones')
