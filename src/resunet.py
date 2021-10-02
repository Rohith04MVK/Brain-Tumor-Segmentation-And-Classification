from tensorflow.keras import layers
from tensorflow.keras.models import Model


def resblock(X, filters):
    """
    Residual blocks
    ---------------
    Residual Blocks are skip-connection blocks that learn residual functions with reference to the layer inputs,
    instead of learning unreferenced functions.
    They were introduced as part of the ResNet architecture.

    Parameters
    ----------
    X
        Input data
    filters
        Number of filters for the convolutional layer
    """
    # Get a copy
    X_copy = X

    # Main path
    X = layers.Conv2D(filters, kernel_size=(1, 1), kernel_initializer="he_normal")(X)
    X = layers.BatchNormalization()(X)
    X = layers.Activation("relu")(X)

    X = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", kernel_initializer="he_normal")(X)
    X = layers.BatchNormalization()(X)

    # Short path
    X_copy = layers.Conv2D(filters, kernel_size=(1, 1), kernel_initializer="he_normal")(X_copy)
    X_copy = layers.BatchNormalization()(X_copy)

    # Combine Main and short path
    X = layers.Add()([X, X_copy])
    X = layers.Activation("relu")(X)

    return X


def upsample_concat(x, skip):
    """
    Funtion for upsampling image

    Parameters
    ----------
    X
        The Input
    skip
        The layer to skip to
    """
    X = layers.UpSampling2D((2, 2))(x)
    return layers.Concatenate()([X, skip])


def resunet(input_shape) -> Model:
    """
    ResUNet
    -------
    A semantic segmentation model inspired by the deep residual learning and UNet.
    An architecture that take advantages from both(Residual and UNet) models.

    Paper: https://arxiv.org/pdf/1711.10684.pdf

    Parameter
    ---------
    input_shape: tuple
        The shape of the data

    Return
    ------
    model: Model
        The ResUNet
    """
    X_input = layers.Input(input_shape)  # iniating tensor of input shape

    # Stage 1
    conv_1 = layers.Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(X_input)
    conv_1 = layers.BatchNormalization()(conv_1)

    conv_1 = layers.Conv2D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)
    pool_1 = layers.MaxPool2D((2, 2))(conv_1)

    # stage 2
    conv_2 = resblock(pool_1, 32)
    pool_2 = layers.MaxPool2D((2, 2))(conv_2)

    # Stage 3
    conv_3 = resblock(pool_2, 64)
    pool_3 = layers.MaxPool2D((2, 2))(conv_3)

    # Stage 4
    conv_4 = resblock(pool_3, 128)
    pool_4 = layers.MaxPool2D((2, 2))(conv_4)

    # Stage 5 (bottle neck)
    conv_5 = resblock(pool_4, 256)

    # Upsample Stage 1
    up_1 = upsample_concat(conv_5, conv_4)
    up_1 = resblock(up_1, 128)

    # Upsample Stage 2
    up_2 = upsample_concat(up_1, conv_3)
    up_2 = resblock(up_2, 64)

    # Upsample Stage 3
    up_3 = upsample_concat(up_2, conv_2)
    up_3 = resblock(up_3, 32)

    # Upsample Stage 4
    up_4 = upsample_concat(up_3, conv_1)
    up_4 = resblock(up_4, 16)

    # final output
    out = layers.Conv2D(1, (1, 1), kernel_initializer="he_normal", padding="same", activation="sigmoid")(up_4)

    return Model(X_input, out)
