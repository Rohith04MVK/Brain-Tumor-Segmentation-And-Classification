import cv2
import numpy as np
import tensorflow as tf
from skimage import io


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        ids,
        mask,
        image_dir="./",
        batch_size=16,
        img_h=256,
        img_w=256,
        shuffle=True,
    ):

        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Get the number of batches per epoch"

        return int(np.floor(len(self.ids)) / self.batch_size)

    def __getitem__(self, index):
        "Generate a batch of data"

        # generate index of batch_size length
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # get the ImageId corresponding to the indexes created above based on batch size
        list_ids = [self.ids[i] for i in indexes]

        # get the MaskId corresponding to the indexes created above based on batch size
        list_mask = [self.mask[i] for i in indexes]

        # generate data for the X(features) and y(label)
        X, y = self.__data_generation(list_ids, list_mask)

        # returning the data
        return X, y

    def on_epoch_end(self):
        "Used for updating the indices after each epoch, once at the beginning as well as at the end of each epoch"

        # getting the array of indices based on the input dataframe
        self.indexes = np.arange(len(self.ids))

        # if shuffle is true, shuffle the indices
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):
        "generate the data corresponding the indexes in a given batch of images"

        # create empty arrays of shape (batch_size,height,width,depth)
        # Depth is 3 for input and depth is taken as 1 for output becasue mask consist only of 1 channel.
        X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
        y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

        # iterate through the dataframe rows, whose size is equal to the batch_size
        for i in range(len(list_ids)):
            # path of the image
            img_path = str(list_ids[i])

            # mask path
            mask_path = str(list_mask[i])

            # reading the original image and the corresponding mask image
            img = io.imread(img_path)
            mask = io.imread(mask_path)

            # resizing and coverting them to array of type float64
            img = cv2.resize(img, (self.img_h, self.img_w))
            img = np.array(img, dtype=np.float64)

            mask = cv2.resize(mask, (self.img_h, self.img_w))
            mask = np.array(mask, dtype=np.float64)

            # standardising
            img -= img.mean()
            img /= img.std()

            mask -= mask.mean()
            mask /= mask.std()

            # Adding image to the empty array
            X[i] = img

            # expanding the dimnesion of the image from (256, 256) to (256, 256, 1)
            y[i] = np.expand_dims(mask, axis=2)

        # normalizing y
        y = (y > 0).astype(int)

        return X, y
