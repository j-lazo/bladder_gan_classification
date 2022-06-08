import tensorflow as tf
from tensorflow import keras
from models.building_blocks import *
from models.model_utils import *

input_sizes_models = {'vgg16': (224, 224), 'vgg19': (224, 224), 'inception_v3': (299, 299),
                      'resnet50': (224, 224), 'resnet101': (224, 244), 'mobilenet': (224, 224),
                      'densenet121': (224, 224), 'xception': (299, 299)}


def build_simple_model_with_domain_input(backbones):
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
    output_layer = Dense(5, activation='softmax')(x)

    return Model(inputs=[input_image, t_input], outputs=output_layer, name='simple_model_with_domain')