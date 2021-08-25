from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


def classification_model(input_size):
    effnet = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=input_size
    )
    model = effnet.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(rate=0.5)(model)
    model = Dense(4, activation="softmax")(model)
    model = Model(inputs=effnet.input, outputs=model)
    return model
