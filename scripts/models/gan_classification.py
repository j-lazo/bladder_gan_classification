from models.building_blocks import *
import tensorflow as tf
from models.model_utils import *
from tensorflow import keras
import os


def build_gan_model_features(backbones=['resnet101', 'resnet101', 'resnet101'], pretrained=True, gan_weights=None,
                             after_concat='globalpooling', train_generators=False):

    input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                          'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                          'densenet121': (224, 224), 'xception': (299, 299)}

    #checkpoint_dir = ''.join([os.getcwd(), '/scripts/gan_models/CycleGan/sample_weights/', gan_base])

    if len(backbones) <= 2:
        backbones = backbones * 3

    # inputs
    input_image = keras.Input(shape=(256, 256, 3), name="image")
    t_input = keras.Input(shape=(1,), name="img_domain")

    # Generator Models
    G_A2B = ResnetGenerator(input_shape=(256, 256, 3))
    G_B2A = ResnetGenerator(input_shape=(256, 256, 3))

    if gan_weights:
        Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), gan_weights).restore()

    num_backbones = len(backbones)

    if train_generators is False:
        for layer in G_A2B.layers:
            layer.trainable = False

        for layer in G_B2A.layers:
            layer.trainable = False

    # branch 1 if target domain is NBI

    c1 = G_A2B(input_image)
    r1 = G_B2A(c1)
    # x1 = [c1, r1]

    # branch 2 if target domain is WLI
    c2 = G_B2A(input_image)
    r2 = G_A2B(c2)
    # x2 = [c2, r2]

    c = tf.keras.backend.switch(t_input, c1, c2)
    r = tf.keras.backend.switch(t_input, r1, r2)

    input_backbone_1 = input_image
    input_backbone_2 = c
    input_backbone_3 = r
    b1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')

    if backbones[0] == 'resnet101':
        b1 = tf.keras.applications.resnet.preprocess_input(b1)
    elif backbones[0] == 'resnet50':
        b1 = tf.keras.applications.resnet50.preprocess_input(b1)
    elif backbones[0] == 'densenet121':
        b1 = tf.keras.applications.densenet.preprocess_input(b1)
    elif backbones[0] == 'vgg19':
        b1 = tf.keras.applications.vgg19.preprocess_input(b1)
    elif backbones[0] == 'inception_v3':
        b1 = tf.keras.applications.inception_v3.preprocess_input(b1)

    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    b1 = backbone_model_1(b1)

    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    if backbones[1] == 'resnet101':
        b2 = tf.keras.applications.resnet.preprocess_input(b2)
    elif backbones[1] == 'resnet50':
        b2 = tf.keras.applications.resnet50.preprocess_input(b2)
    elif backbones[1] == 'densenet121':
        b2 = tf.keras.applications.densenet.preprocess_input(b2)
    elif backbones[1] == 'vgg19':
        b2 = tf.keras.applications.vgg19.preprocess_input(b2)
    elif backbones[1] == 'inception_v3':
        b2 = tf.keras.applications.inception_v3.preprocess_input(b2)
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False
    b2 = backbone_model_2(b2)
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        if backbones[2] == 'resnet101':
            b3 = tf.keras.applications.resnet.preprocess_input(b3)
        elif backbones[2] == 'resnet50':
            b3 = tf.keras.applications.resnet50.preprocess_input(b3)
        elif backbones[2] == 'densenet121':
            b3 = tf.keras.applications.densenet.preprocess_input(b3)
        elif backbones[2] == 'vgg19':
            b3 = tf.keras.applications.vgg19.preprocess_input(b3)
        elif backbones[2] == 'inception_v3':
            b3 = tf.keras.applications.inception_v3.preprocess_input(b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])
    if after_concat == 'globalpooling':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_merge_classification')