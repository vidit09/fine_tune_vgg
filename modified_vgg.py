from keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model


def get_model(num_class):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    last_conv = vgg.output

    x = Flatten()(last_conv)
    x = Dense(256, activation='relu')(x)#base_version
    # x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation='softmax')(x)

    model = Model(vgg.input, x)

    for layer in model.layers[:15]:
        print(layer)
        layer.trainable = False

    return model