import os
import numpy as np
import pandas as pd
import cv2
from glob import glob

import tensorflow as tf
from keras.layers import *
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    path = "Dog Breed Identification/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []


#Error here when reading labels.
    for image_id in ids:
        if list(labels_df[labels_df.id== image_id]["breed"]) == 0:
            print(image_id)
            continue
        image_id = image_id.split("/")[-1].split(".")[0]
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        print(image_id, breed_name)