import tensorflow as tf
from tensorflow import keras

from src.losses import focal_tversky, tversky
from src.predict import prediction
from src.resunet import resblock, upsample_concat
from src.train_seg import test

clf_model = keras.models.load_model("./models/seg_cls_res.hdf5")

seg_model = tf.keras.models.load_model(
    "./models/ResUNet-weights.hdf5",
    custom_objects={
        "resblock": resblock,
        "upsample_concat": upsample_concat,
        "focal_tversky": focal_tversky,
        "tversky": tversky,
    },
)

df_pred = prediction(test, clf_model, seg_model)
print(df_pred.head())
