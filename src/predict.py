import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io
from tensorflow.keras.models import load_model


def prediction(test, model, model_seg):
    """
    Pipeline to connect the classification and segmentation model.
    All the preprocessing and detection takes place in this function.
    The classification model first classifies the image as have or not having a tumor
    if the classificaiton model is 99% sure there is no tumor the imgae is marked as no tumor.
    If it's not sure the image is given to the segmentation model which then predicts the the pixel area of the tumor.
    """
    # empty list to store results
    mask, image_id, has_mask = [], [], []

    # itetrating through each image in test data
    for i in test.image_path:

        img = io.imread(i)
        # normalizing
        img = img * 1.0 / 255.0
        # reshaping
        img = cv2.resize(img, (256, 256))
        # converting img into array
        img = np.array(img, dtype=np.float64)
        # reshaping the image from 256,256,3 to 1,256,256,3
        img = np.reshape(img, (1, 256, 256, 3))

        # making prediction for tumor in image
        is_defect = model.predict(img)

        # if tumour is not present we append the details of the image to the list
        if np.argmax(is_defect) == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append("No mask.")
            continue

        # Creating a empty array of shape 1,256,256,1
        X = np.empty((1, 256, 256, 3))
        # read the image
        img = io.imread(i)
        # resizing the image and coverting them to array of type float64
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        # standardising the image
        img -= img.mean()
        img /= img.std()
        # converting the shape of image from 256,256,3 to 1,256,256,3
        X[
            0,
        ] = img

        # make prediction of mask
        predict = model_seg.predict(X)

        # if sum of predicted mask is 0 then there is not tumour
        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append("No mask :)")
        else:
            # if the sum of pixel values are more than 0, then there is tumour
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return pd.DataFrame(
        {"image_path": image_id, "predicted_mask": mask, "has_mask": has_mask}
    )
