import math
import os
import random

import numpy as np
from tensorflow.keras.utils import Sequence

from utils.image_utils import load_image

class DataSequence(Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, data_root, batch_size, resize_size=(64, 64), flip_augment=True, mode='train'):
        self.df = df
        self.batch_size = batch_size
        self.mode = mode
        self.resize_size = resize_size
        self.crop_pt_1 = (45, 25)
        self.crop_pt_2 = (173, 153)
        self.flip_augment = flip_augment
        # extract columns from df columns
        self.label_columns = self.df.columns[1:].tolist() 

        # Take labels and a list of image locations in memory
        self.labels = self.df[self.label_columns].values
        self.im_list = self.df['Image_Name'].apply(lambda x: os.path.join(data_root, x)).tolist()
        # Trigger a shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(math.floor(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return self.labels[idx]

    def get_batch_features(self, idx):
        images = []
        for im_idx in idx:
            im = self.im_list[im_idx]
            loaded_image = load_image(im, self.resize_size, self.crop_pt_1, self.crop_pt_2)
            if self.flip_augment and random.random() < 0.5:
                loaded_image = np.flip(loaded_image, 1)
            images.append(loaded_image)
        # Fetch a batch of inputs
        return np.array(images)

    def __getitem__(self, index):
        idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # get the actual data
        batch_x = self.get_batch_features(idx)
        batch_y = np.clip(self.get_batch_labels(idx).astype(np.float32), 0, 1)
        return (batch_x, batch_y), batch_y
