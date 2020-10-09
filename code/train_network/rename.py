"""
This script renames the name of the folders for the generated images dataset.
"""
import os
os.chdir('/home/yuqi/imagenet_replay/val_resized')
list_of_dirs_val = os.listdir()
list_of_dirs_val.sort()

os.chdir('/home/yuqi/imagenet_replay/train')  # The path of the generated images dataset
for imagenet_class in range(1000):
    oldname = str(imagenet_class)
    newname = list_of_dirs_val[imagenet_class]
    os.rename(oldname, newname)