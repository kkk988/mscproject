"""
This script is for deleting the images that are more than 1000 images in one class.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

image_base_dir = '/home/yuqi/generate_images/generated_images_colour_check'

for imagenet_class in range(1000):
    image_dir = os.path.join(image_base_dir,str(imagenet_class))
    os.chdir(image_dir)
    file_list = os.listdir()
    file_list.sort()
    count = 0
    for filename in file_list:
        count += 1
        if count > 1000:
            os.remove(filename)
