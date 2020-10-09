"""
This script takes all the new generated images, puts them in a new folder,
selects the ones which are top 5, and deletes the rest
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")

from distutils.dir_util import copy_tree
# Copy all the generated images to a new folder
fromDirectory = "/home/yuqi/generate_images/generated_images_colour"
toDirectory = "/home/yuqi/generate_images/generated_images_colour_check"
copy_tree(fromDirectory,toDirectory) 

from os import walk
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import imageio

# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
shape = 224,224,3
top_5 = 0
not_top_5 = 0
image_base_dir = '/home/yuqi/generate_images/generated_images_colour_check'
top_5_by_class = np.zeros(1000)
not_top_5_by_class = np.zeros(1000)

# This part deletes corrupt files
for imagenet_class in range(1000):
	image_dir = os.path.join(image_base_dir,str(imagenet_class))
	os.chdir(image_dir)
	file_list = os.listdir()
	file_list.sort()

	for filename in file_list:
		try:
			image_corrupt = imageio.imread(filename)  # Open the image file
		except:
			print('Bad file:', filename)
			os.remove(filename)  # Delete the corrupt file

# This part deletes the images that are not top 5 image
for imagenet_class in range(1000):
	image_dir = os.path.join(image_base_dir,str(imagenet_class))
	os.chdir(image_dir)
	file_list = os.listdir()
	file_list.sort()
	file_count = 0
	for filename in file_list:
		# Load an image from file
		image = load_img(filename, target_size=(224, 224))
		# Convert the image pixels to a numpy array
		image = img_to_array(image)
		image = img_to_array(image)
		image = preprocess_input(image)
		# Predict the probability across all output classes
		image = np.expand_dims(image, axis=0)
		yhat = model.predict(image)
		yhat2 = np.squeeze(yhat)
		yhat3 = [np.argsort(yhat2)[-5:]]
		if np.isin(imagenet_class,yhat3):  # Check whether the image class is in the top 5 predictions
			top_5 = top_5+1
			top_5_by_class[imagenet_class] = top_5_by_class[imagenet_class] + 1  # Record the number of top 5 images in one class
			print('top_5',top_5,'not_top_5',not_top_5)
			file_count = file_count + 1
		else:
			# If not top 5, delete them
			not_top_5 = not_top_5 + 1
			not_top_5_by_class[imagenet_class] = not_top_5_by_class[imagenet_class] + 1
			print('top_5',top_5,'not_top_5',not_top_5)
			os.remove(filename)

percentage_top_5 = (top_5/(len(file_list))*100)

# This part keeps a record of how many images in each class that are top 5 or not.
# We can then inspect this to see how many more we need to generate.
# We can see the names of the classes see here:
# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
np.savetxt("/home/yuqi/generate_images/check/top_5_by_class.csv", top_5_by_class, delimiter=",")
np.savetxt("/home/yuqi/generate_images/check/not_top_5_by_class.csv", not_top_5_by_class, delimiter=",")
np.save('/home/yuqi/generate_images/check/top_5_by_class.npy',top_5_by_class)
np.save('/home/yuqi/generate_images/check/not_top_5_by_class.npy',not_top_5_by_class)