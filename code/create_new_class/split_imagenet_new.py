"""
This script splits the 1000 images of new class into 90% training data and 10% validation data
"""
import splitfolders  # or import split_folders

# Split with a ratio
splitfolders.ratio("/home/yuqi/data_set/new_class", output="/home/yuqi/data_set/new_class_split", seed=1337, ratio=(.9, .1), group_prefix=None) # default values
