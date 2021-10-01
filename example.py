import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
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

df_pred = prediction(test, clf_model, seg_model)
print(df_pred.head())
