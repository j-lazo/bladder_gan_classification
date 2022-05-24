import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.building_blocks import *
from models.model_utils import *

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299)}


def build_gan_model_features(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
                             after_concat='globalpooling'):

    if len(backbones) <= 2:
        backbones = backbones * 3

    num_backbones = len(backbones)

    # inputs
    input_image = keras.Input(shape=(256, 256, 3), name="image")
    t_input = keras.Input(shape=(1,), name="img_domain")

    # Generator Models
    G_A2B = ResnetGenerator(input_shape=(256, 256, 3))
    G_B2A = ResnetGenerator(input_shape=(256, 256, 3))

    if gan_weights:
        Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), gan_weights).restore()

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

    # outputs from the generators
    c = tf.keras.backend.switch(t_input, c1, c2)
    r = tf.keras.backend.switch(t_input, r1, r2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(c, tf.reduce_min(c)),
                                                 tf.math.subtract(tf.reduce_max(c), tf.math.reduce_min(c))))
    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(r, tf.reduce_min(r)),
                                                 tf.math.subtract(tf.reduce_max(r), tf.math.reduce_min(r))))

    input_backbone_2 = tf.cast(input_backbone_2, tf.float32)
    input_backbone_3 = tf.cast(input_backbone_3, tf.float32)

    #input_backbone_2 = c
    #input_backbone_3 = r

    # load the backbones
    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'
    for layer in backbone_model_1.layers:
        layer.trainable = False
    backbone_model_2 = load_pretrained_backbones(backbones[1])
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = False

    # branch 1 takes the original image
    b1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')
    b1 = get_preprocess_input_backbone(backbones[0], b1)
    b1 = backbone_model_1(b1)

    # branch 2 takes the image created by the first generator
    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    b2 = get_preprocess_input_backbone(backbones[1], b2)
    b2 = backbone_model_2(b2)

    # branch 3 takes the image created by the second generators
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        b3 = get_preprocess_input_backbone(backbones[1], b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        # concatenate the feature maps from the 3 backbones
        x = Concatenate()([b1, b2, b3])

    else:
        # concatenate the feature maps from the backbones
        x = Concatenate()([b1, b2])

    if after_concat == 'globalpooling':
        x = GlobalAveragePooling2D()(x)
    else:
        x = Flatten()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_merge_classification')