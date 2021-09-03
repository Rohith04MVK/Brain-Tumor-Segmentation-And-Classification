import tensorflow as tf
import tensorflow.keras.backend as K

epsilon = 1e-5
smooth = 1


def tversky(y_true, y_pred):
    """
    The Tversky Index (TI) is a asymmetric similarity measure that is a
    generalisation of the dice coefficient and the Jaccard index.

    Parameters
    ----------
    y_ture - The true values oe the labels
    y_pred - The predicted labels
    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth
    )


def focal_tversky(y_true, y_pred):
    """
    The Focal Tversky Loss (FTL) is a generalisation of the tversky loss.
    The non-linear nature of the loss gives you control over how the loss behaves at different values of the tversky index obtained.
    γ is a parameter that controls the non-linearity of the loss.

    Parameters
    ----------
    y_ture - The true values oe the labels
    y_pred - The predicted labels
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def tversky_loss(y_true, y_pred):
    """
    It's just the Tversky Index subtracted from 1

    Parameters
    ----------
    y_ture - The true values oe the labels
    y_pred - The predicted labels
    """
    return 1 - tversky(y_true, y_pred)
