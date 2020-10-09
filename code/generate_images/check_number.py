"""
This script check whether the number of images in each class is 1000,
to make sure our new generated images dataset contains 1000 images for each class.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
image_base_dir = '/home/yuqi/generate_images/generated_images_colour_check'

for imagenet_class in range(1000):
    image_dir = os.path.join(image_base_dir,str(imagenet_class))
    os.chdir(image_dir)
    file_list = os.listdir()
    file_list.sort()
    total_num = len(file_list)
    if total_num != 1000:
        print(imagenet_class)
