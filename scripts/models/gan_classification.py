import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.building_blocks import *
from models.model_utils import *
from models.simple_models import *

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299),
                      'simple_residual_model': (256, 256)}


def build_gan_model_joint_features(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
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
    # if you want to have a classifiers evaluating the results from converted and re-converted uncomment bellow
    """gen_a2b = tf.keras.backend.switch(t_input, c1, c2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, r2)"""

    # if you want to have a classifiers evaluating the results produced by each Generator separate use bellow
    gen_a2b = tf.keras.backend.switch(t_input, c1, r2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, c2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_a2b, tf.reduce_min(gen_a2b)),
                                                               tf.math.subtract(tf.reduce_max(gen_a2b),
                                                                                tf.math.reduce_min(gen_a2b))))

    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_b2a, tf.reduce_min(gen_b2a)),
                                                               tf.math.subtract(tf.reduce_max(gen_b2a),
                                                                                tf.math.reduce_min(gen_b2a))))

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

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_merge_features')


def build_gan_model_joint_features_and_domain(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
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
    # if you want to have a classifiers evaluating the results from converted and re-converted uncomment bellow
    """gen_a2b = tf.keras.backend.switch(t_input, c1, c2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, r2)"""

    # if you want to have a classifiers evaluating the results produced by each Generator separate use bellow
    gen_a2b = tf.keras.backend.switch(t_input, c1, r2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, c2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_a2b, tf.reduce_min(gen_a2b)),
                                                               tf.math.subtract(tf.reduce_max(gen_a2b),
                                                                                tf.math.reduce_min(gen_a2b))))

    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_b2a, tf.reduce_min(gen_b2a)),
                                                               tf.math.subtract(tf.reduce_max(gen_b2a),
                                                                                tf.math.reduce_min(gen_b2a))))

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

    x = Concatenate()([x, t_input])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Flatten()(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_merge_features_and_domain')


def build_gan_model_separate_features(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
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
    # if you want to have a classifiers evaluating the results from converted and re-converted uncomment bellow
    """gen_a2b = tf.keras.backend.switch(t_input, c1, c2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, r2)"""

    # if you want to have a classifiers evaluating the results produced by each Generator separate use bellow
    gen_a2b = tf.keras.backend.switch(t_input, c1, r2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, c2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_a2b, tf.reduce_min(gen_a2b)),
                                                               tf.math.subtract(tf.reduce_max(gen_a2b),
                                                                                tf.math.reduce_min(gen_a2b))))

    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_b2a, tf.reduce_min(gen_b2a)),
                                                               tf.math.subtract(tf.reduce_max(gen_b2a),
                                                                                tf.math.reduce_min(gen_b2a))))

    input_backbone_2 = tf.cast(input_backbone_2, tf.float32)
    input_backbone_3 = tf.cast(input_backbone_3, tf.float32)

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
    b1 = GlobalAveragePooling2D()(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(512, activation='relu')(b1)
    b1 = Flatten()(b1)

    # branch 2 takes the image created by the first generator
    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    b2 = get_preprocess_input_backbone(backbones[1], b2)
    b2 = backbone_model_2(b2)
    b2 = GlobalAveragePooling2D()(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(512, activation='relu')(b2)
    b2 = Flatten()(b2)

    # branch 3 takes the image created by the second generators
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        b3 = get_preprocess_input_backbone(backbones[1], b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        # concatenate the the 3 branches
        b3 = GlobalAveragePooling2D()(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(512, activation='relu')(b3)
        b3 = Flatten()(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_separate_features')


def build_gan_model_separate_features_v2(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
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
    # if you want to have a classifiers evaluating the results from converted and re-converted uncomment bellow
    """gen_a2b = tf.keras.backend.switch(t_input, c1, c2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, r2)"""

    # if you want to have a classifiers evaluating the results produced by each Generator separate use bellow
    gen_a2b = tf.keras.backend.switch(t_input, c1, r2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, c2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_a2b, tf.reduce_min(gen_a2b)),
                                                               tf.math.subtract(tf.reduce_max(gen_a2b),
                                                                                tf.math.reduce_min(gen_a2b))))

    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_b2a, tf.reduce_min(gen_b2a)),
                                                               tf.math.subtract(tf.reduce_max(gen_b2a),
                                                                                tf.math.reduce_min(gen_b2a))))

    input_backbone_2 = tf.cast(input_backbone_2, tf.float32)
    input_backbone_3 = tf.cast(input_backbone_3, tf.float32)

    # load the backbones
    backbone_model_1 = load_pretrained_backbones(backbones[0])
    backbone_model_1._name = 'backbone_1'

    for layer in backbone_model_1.layers:
        layer.trainable = False
    backbone_model_2 = simple_residual_model()
    backbone_model_2._name = 'backbone_2'
    for layer in backbone_model_2.layers:
        layer.trainable = True

    # branch 1 takes the original image
    b1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')
    b1 = get_preprocess_input_backbone(backbones[0], b1)
    b1 = backbone_model_1(b1)
    b1 = GlobalAveragePooling2D()(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(512, activation='relu')(b1)
    b1 = Flatten()(b1)

    # branch 2 takes the image created by the first generator
    b2 = tf.image.resize(input_backbone_2, input_sizes_models['simple_residual_model'], method='bilinear')/255.
    b2 = backbone_model_2(b2)
    b2 = GlobalAveragePooling2D()(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(512, activation='relu')(b2)
    b2 = Flatten()(b2)

    # branch 3 takes the image created by the second generators
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        b3 = get_preprocess_input_backbone(backbones[1], b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        # concatenate the the 3 branches
        b3 = GlobalAveragePooling2D()(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(512, activation='relu')(b3)
        b3 = Flatten()(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_separate_features')


def dimage_domain_classifier(x, weights=None):
    pass
    return 0 #Model()


def build_gan_model_joint_features_plus_domain(backbones=['resnet101', 'resnet101', 'resnet101'], gan_weights=None,
                             after_concat='globalpooling'):


    if len(backbones) <= 2:
        backbones = backbones * 3

    num_backbones = len(backbones)

    # inputs
    input_image = keras.Input(shape=(256, 256, 3), name="image")

    d1 = tf.image.resize(input_image, input_sizes_models[backbones[0]], method='bilinear')
    d1 = get_preprocess_input_backbone(backbones[0], d1)
    domain_classifier = backbone_model_1(d1)


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
    # if you want to have a classifiers evaluating the results from converted and re-converted uncomment bellow
    """gen_a2b = tf.keras.backend.switch(t_input, c1, c2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, r2)"""

    # if you want to have a classifiers evaluating the results produced by each Generator separate use bellow
    gen_a2b = tf.keras.backend.switch(t_input, c1, r2)
    gen_b2a = tf.keras.backend.switch(t_input, r1, c2)

    # then we need to resize and re-scale the data
    input_backbone_2 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_a2b, tf.reduce_min(gen_a2b)),
                                                 tf.math.subtract(tf.reduce_max(gen_a2b), tf.math.reduce_min(gen_a2b))))

    input_backbone_3 = tf.math.scalar_mul(255., tf.math.divide(tf.math.subtract(gen_b2a, tf.reduce_min(gen_b2a)),
                                                 tf.math.subtract(tf.reduce_max(gen_b2a), tf.math.reduce_min(gen_b2a))))

    input_backbone_2 = tf.cast(input_backbone_2, tf.float32)
    input_backbone_3 = tf.cast(input_backbone_3, tf.float32)

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
    b1 = GlobalAveragePooling2D()(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(1024, activation='relu')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Dense(512, activation='relu')(b1)
    b1 = Flatten()(b1)

    # branch 2 takes the image created by the first generator
    b2 = tf.image.resize(input_backbone_2, input_sizes_models[backbones[1]], method='bilinear')
    b2 = get_preprocess_input_backbone(backbones[1], b2)
    b2 = backbone_model_2(b2)
    b2 = GlobalAveragePooling2D()(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(1024, activation='relu')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Dense(512, activation='relu')(b2)
    b2 = Flatten()(b2)

    # branch 3 takes the image created by the second generators
    if num_backbones == 3:
        b3 = tf.image.resize(input_backbone_3, input_sizes_models[backbones[2]], method='bilinear')
        b3 = get_preprocess_input_backbone(backbones[1], b3)
        backbone_model_3 = load_pretrained_backbones(backbones[2])
        backbone_model_3._name = 'backbone_3'
        for layer in backbone_model_3.layers:
            layer.trainable = False
        b3 = backbone_model_3(b3)
        # concatenate the the 3 branches
        b3 = GlobalAveragePooling2D()(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(1024, activation='relu')(b3)
        b3 = Dropout(0.5)(b3)
        b3 = Dense(512, activation='relu')(b3)
        b3 = Flatten()(b3)
        x = Concatenate()([b1, b2, b3])

    else:
        x = Concatenate()([b1, b2])

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='gan_separate_features')