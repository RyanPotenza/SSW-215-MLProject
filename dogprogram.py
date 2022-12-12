import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

dataset, dsinfo = tfds.load('stanford_dogs', with_info=True)
label_name = dsinfo.features['label'].int2str

for i in dataset['train'].take(10):
    plt.figure()
    plt.imshow(i['image'])
    plt.title(label_name(i['label']))
    
    
