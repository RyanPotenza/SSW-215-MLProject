import tensorflow as tf
import tensorflow_datasets as tfds

train_ds, test_ds = tfds.load('stanford_dogs', split=['train', 'test[:10%]'])
