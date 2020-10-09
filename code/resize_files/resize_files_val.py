"""
This script resizes the files in the validation set in our project.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Copy the validation set to a new path
from distutils.dir_util import copy_tree
fromDirectory = "/fast-data20/datasets/daniel/0_imagenet_split/val"
toDirectory = "/home/yuqi/imagenet_replay/val_resized"
copy_tree(fromDirectory, toDirectory)

# This part is for resizing
import glob

my_path = "/home/yuqi/imagenet_replay/val_resized"
files = glob.glob(my_path + '/**/*.JPEG', recursive=True)

from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
from skimage import img_as_ubyte

for val_file in range(len(files)):
    img = imread(files[val_file])
    img = resize(img, (224, 224), anti_aliasing=True)
    imsave(files[val_file], img_as_ubyte(img))



