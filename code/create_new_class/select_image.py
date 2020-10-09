"""
This script selects one image from the generated images dataset for each class.
"""
import os
import numpy as np
from shutil import copyfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Generate a random number for 1000 classes, as the index of the selected image
# in each class
index_list = np.random.randint(0,1000,size=1000)

# The path of the generated images dataset
image_base_dir = '/home/yuqi/new_class/train'
# The path of the folder that have selected images for each class
new_class_dir = '/home/yuqi/new_class/new_class'

for imagenet_class in range(1000):
    image_dir = os.path.join(image_base_dir,str(imagenet_class))
    os.chdir(image_dir)
    file_list = os.listdir()
    file_list.sort()
    filename = file_list[index_list[imagenet_class]]  # Select the image for one class
    source = os.path.join(image_base_dir,str(imagenet_class),filename)
    target_dic = os.path.join(new_class_dir,str(imagenet_class))
    if not os.path.exists(target_dic):
        os.mkdir(target_dic)
    target = os.path.join(target_dic,filename)
    copyfile(source, target)  # Copy the selected images to a new folder