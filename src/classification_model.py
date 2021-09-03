from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model


def classification_model(input_size) -> Model:
    """
    The model used to classify the type of tumor.
    This function uses the `ResNet50` for transfer learning.

    Parameters
    ----------
    input_size - The size of the input data

    Return
    ------
    model - tensorflow.keras.models.Model including the `ResNet50` weights with additional layers for the classification
    """
    clf_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=Input(input_size)
    )
    model = clf_model.output
    model = AveragePooling2D(pool_size=(4, 4))(model)
    model = Flatten(name="Flatten")(model)
    model = Dense(256, activation="relu")(model)
    model = Dropout(0.3)(model)
    model = Dense(256, activation="relu")(model)
    model = Dropout(0.3)(model)
    model = Dense(2, activation="softmax")(model)

    model = Model(clf_model.input, model)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model
