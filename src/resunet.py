import tensorflow as tf
from tensorflow.keras.layers import *


def resblock(X, filters):
    """
    Residual blocks
    ---------------
    Residual Blocks are skip-connection blocks that learn residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.
    They were introduced as part of the ResNet architecture.

    Parameters
    ----------
    X - Input data

    filters - Number of filters for the convolutional layer

    Return
    ------

    """
    X_copy = X  # copy of input

    # main path
    X = Conv2D(filters, kernel_size=(1, 1), kernel_initializer="he_normal")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)

    X = Conv2D(
        filters, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal"
    )(X)
    X = BatchNormalization()(X)

    # shortcut path
    X_copy = Conv2D(filters, kernel_size=(1, 1), kernel_initializer="he_normal")(X_copy)
    X_copy = BatchNormalization()(X_copy)

    # Adding the output from main path and short path together
    X = Add()([X, X_copy])
    X = Activation("relu")(X)

    return X


def upsample_concat(x, skip):
    """
    funtion for upsampling image
    """
    X = UpSampling2D((2, 2))(x)
    return Concatenate()([X, skip])
