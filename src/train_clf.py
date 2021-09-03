import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from classification_model import classification_model

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

print(brain_df["mask"].value_counts())

brain_df_train = brain_df.drop(columns=["patient_id"])
# Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
brain_df_train["mask"] = brain_df_train["mask"].apply(lambda x: str(x))
print(brain_df_train.info())


train, test = train_test_split(brain_df_train, test_size=0.15)

datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1)

train_generator = datagen.flow_from_dataframe(
    train,
    directory="./",
    x_col="image_path",
    y_col="mask",
    subset="training",
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    target_size=(256, 256),
)
valid_generator = datagen.flow_from_dataframe(
    train,
    directory="./",
    x_col="image_path",
    y_col="mask",
    subset="validation",
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    target_size=(256, 256),
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_dataframe(
    test,
    directory="./",
    x_col="image_path",
    y_col="mask",
    class_mode="categorical",
    batch_size=16,
    shuffle=False,
    target_size=(256, 256),
)

earlystopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)
checkpointer = ModelCheckpoint(
    filepath="./models/seg_cls_res.hdf5",
    verbose=1,
    save_best_only=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", mode="min", verbose=1, patience=10, min_delta=0.0001, factor=0.2
)
callbacks = [checkpointer, earlystopping, reduce_lr]

model = classification_model(input_size=(256, 256, 3))
print("Model loaded!")

h = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=50,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    callbacks=[checkpointer, earlystopping],
)
