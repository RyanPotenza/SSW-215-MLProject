import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

# train_ds, test_ds = tfds.load('stanford_dogs', split=['train', 'test[:10%]'])
dataset = tfds.builder('stanford_dogs')
info = dataset.info
info.features
class_names = []
for i in range(info.features["label"].num_classes):
    class_names.append(info.features["label"].int2str(i))
class_names
