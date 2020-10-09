"""
This script shows the distribution of predictions of the images in the test set in Model 1.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# screen check_top_5
from os import walk
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import imageio
from tensorflow.keras.models import load_model

# Load the best model we saved when training the Model 1
model = load_model('/home/yuqi/imagenet_replay/checkpoint/vgg_16_train_from_scratch/saved_model.h5', custom_objects=None, compile=True)
shape = 224,224,3
image_base_dir = '/home/yuqi/imagenet_replay/test_resized'
top_5_by_class_array = np.zeros(1001)
total_top_5 = 0
total_file_count = 0

pred_dic = {}  # Used for saving the results of predictions

for imagenet_class in range(1000):
    print(imagenet_class)
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
        features = model.predict(image)
        pred = decode_predictions(features, top=1)[0]
        for element in pred:
            if element[1] not in pred_dic:
                pred_dic[element[1]] = 1
            else:
                pred_dic[element[1]] = pred_dic[element[1]] + 1

# This part delete the times of the classes predicted by Model 1 
# that are small than 50
del_list = []
for label in pred_dic:
    if pred_dic[label] < 50:
        del_list.append(label)

for name in del_list:
    del pred_dic[name]


# This part plots the distributions of predictions
RANGE = 1000
by_value = sorted(pred_dic.items(), key=lambda item: item[1], reverse=True)

x = []
y = []
for d in by_value:
    x.append(d[0])
    y.append(d[1])

print(len(x))


os.chdir('/home/yuqi/imagenet_replay/testing/distribution/')
fig = plt.figure(figsize=(8,15))
ax = fig.add_subplot(111)
ax.barh(x[0:RANGE], y[0:RANGE], fc='cornflowerblue')
ax.set_title(u'Distribution of predictions')
fig.subplots_adjust(left=0.2)
fig = plt.gcf()
name = 'distribution'
fig.savefig(name, dpi=500)

