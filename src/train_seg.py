import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_generator import DataGenerator
from losses import focal_tversky, tversky
from resunet import resunet

data = pd.read_csv("./lgg-mri-segmentation/kaggle_3m/data.csv")

data_map = []
for sub_dir_path in glob.glob("./lgg-mri-segmentation/kaggle_3m/" + "*"):
    # if os.path.isdir(sub_path_dir):
    try:
        dir_name = sub_dir_path.split("/")[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + "/" + filename
            data_map.extend([dir_name, image_path])
    except Exception as e:
        print(e)

df = pd.DataFrame({"patient_id": data_map[::2], "path": data_map[1::2]})

df_imgs = df[~df["path"].str.contains("mask")]
df_masks = df[df["path"].str.contains("mask")]


# Data sorting
imgs = sorted(
    df_imgs["path"].values, key=lambda x: int(re.search(r"\d+", x[-7:]).group())
)
masks = sorted(
    df_masks["path"].values, key=lambda x: int(re.search(r"\d+", x[-12:]).group())
)

# Sorting check
idx = random.randint(0, len(imgs) - 1)
print("Path to the Image:", imgs[idx], "\nPath to the Mask:", masks[idx])

# Final dataframe
brain_df = pd.DataFrame(
    {"patient_id": df_imgs.patient_id.values, "image_path": imgs, "mask_path": masks}
)


def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


brain_df["mask"] = brain_df["mask_path"].apply(lambda x: pos_neg_diagnosis(x))

brain_df_train = brain_df.drop(columns=["patient_id"])
# Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
brain_df_train["mask"] = brain_df_train["mask"].apply(lambda x: str(x))

train, test = train_test_split(brain_df_train, test_size=0.15)

brain_df_mask = brain_df[brain_df["mask"] == 1]
print(brain_df_mask.shape)

# creating test, train and val sets
X_train, X_val = train_test_split(brain_df_mask, test_size=0.15)
X_test, X_val = train_test_split(X_val, test_size=0.5)
print(
    "Train size is {}, valid size is {} & test size is {}".format(
        len(X_train), len(X_val), len(X_test)
    )
)

train_ids = list(X_train.image_path)
train_mask = list(X_train.mask_path)

val_ids = list(X_val.image_path)
val_mask = list(X_val.mask_path)

train_data = DataGenerator(train_ids, train_mask)
val_data = DataGenerator(val_ids, val_mask)

seg_model = resunet(input_shape=(256, 256, 3))

adam = tf.keras.optimizers.Adam(learning_rate=0.05, epsilon=0.1)

earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
# save the best model with lower validation loss
checkpointer = ModelCheckpoint(
    filepath="./models/ResUNet-segModel-weights.hdf5",
    verbose=1,
    save_best_only=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.0001, factor=0.2
)

seg_model.compile(optimizer=adam, loss=focal_tversky, metrics=[tversky])

history = seg_model.fit(
    train_data,
    epochs=60,
    validation_data=val_data,
    callbacks=[checkpointer, earlystopping, reduce_lr],
)
