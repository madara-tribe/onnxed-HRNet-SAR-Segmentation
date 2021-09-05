import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate
import tensorflow_addons as tfa


def transition_layer1(x, out_filters_list=[32, 64]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization()(x0)
    #x0 = Activation('relu')(x0)
    x0 = tfa.activations.mish(x0)
    x1 = Conv2D(out_filters_list[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    x1 = tfa.activations.mish(x1)
    return [x0, x1]

def transition_layer2(x, out_filters_list=[32, 64, 128]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization()(x0)
    #x0 = Activation('relu')(x0)
    x0 = tfa.activations.mish(x0)
    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    x1 = tfa.activations.mish(x1)
    x2 = Conv2D(out_filters_list[2], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization()(x2)
    #x2 = Activation('relu')(x2)
    x2 = tfa.activations.mish(x2)
    return [x0, x1, x2]


def transition_layer3(x, out_filters_list=[32, 64, 128, 256]):
    x0 = Conv2D(out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x0 = BatchNormalization()(x0)
    #x0 = Activation('relu')(x0)
    x0 = tfa.activations.mish(x0)
    
    x1 = Conv2D(out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    x1 = tfa.activations.mish(x1)

    x2 = Conv2D(out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x2 = BatchNormalization()(x2)
    #x2 = Activation('relu')(x2)
    x2 = tfa.activations.mish(x2)
    
    x3 = Conv2D(out_filters_list[3], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization()(x3)
    #x3 = Activation('relu')(x3)
    x3 = tfa.activations.mish(x3)
    return [x0, x1, x2, x3]


