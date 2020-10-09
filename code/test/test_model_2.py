"""
This script tests the top 5 accuracy for Model 2.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from os import walk
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import imageio
from tensorflow.keras.models import load_model

# Load the best model we saved when training the Model 2
model = load_model('/home/yuqi/imagenet_replay/checkpoint/vgg_16_train_imagenet/saved_model.h5', custom_objects=None, compile=True)
shape = 224,224,3
image_base_dir = '/home/yuqi/imagenet_replay/test_resized'

# Create an array to record the top 5 accuracy for each class and the overall top 5 accuracy
top_5_by_class_array=np.zeros(1001)
total_top_5 = 0
total_file_count = 0


for imagenet_class in range(1000):
    image_dir = os.path.join(image_base_dir,str(imagenet_class))
    os.chdir(image_dir)
    file_list = os.listdir()
    file_list.sort()
    top_5_by_class = 0

    for filename in file_list:
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        yhat = model.predict(image)
        yhat2 = np.squeeze(yhat)
        yhat3 = [np.argsort(yhat2)[-5:]]
        if np.isin(imagenet_class,yhat3):
            top_5_by_class = top_5_by_class + 1

    # Calculate the top 5 accuracy for each class 
    percentage_top_5_by_class = top_5_by_class/(len(file_list))*100
    top_5_by_class_array[imagenet_class] = percentage_top_5_by_class  # Record the top 5 accuracy for this class
    print('Class:', imagenet_class, percentage_top_5_by_class)
    total_top_5 = total_top_5 + top_5_by_class
    total_file_count = total_file_count + len(file_list)

# Calculate the top 5 accuracy for the whole test set
top_5_accuracy = total_top_5/total_file_count*100
print('Total:', top_5_accuracy)
top_5_by_class_array[1000] = top_5_accuracy

np.savetxt("/home/yuqi/imagenet_replay/testing/top_5_prediction/fine_tune_top_5_by_class.csv", top_5_by_class_array, delimiter=",")

