import os
import sys
import numpy as np

import tensorflow as tf
keras = tf.keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

model = keras.models.load_model("dogs_vs_cats.h5")

for local_path in os.listdir("TestingImages"):
  img_path = os.path.join(sys.path[0], f"TestingImages/{local_path}")
  img = image.load_img(img_path, target_size=(160, 160))
  img_array = image.img_to_array(img)
  img_batch = np.expand_dims(img_array, axis=0)
  img_preprocessed = preprocess_input(img_batch)
  prediction = model.predict(img_preprocessed)
  if prediction[0] >= 0:
    print(f"{local_path} is an image of a dog")
  else:
    print(f"{local_path} is an image of a cat")