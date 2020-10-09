"""
This script copies the selected images on the class-specific folder
into a new folder
"""
import os
from shutil import copyfile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_base_dir = '/home/yuqi/new_class/new_class'
new_class = 1000  # The name of the new folder for the new class

for imagenet_class in range(1000):
    image_dir = os.path.join(image_base_dir,str(imagenet_class))
    os.chdir(image_dir)
    file_list = os.listdir()
    file_list.sort()
    filename = file_list[0]
    source = os.path.join(image_base_dir,str(imagenet_class),filename)
    target_dic = os.path.join(image_base_dir,str(new_class))
    if not os.path.exists(target_dic):
        os.mkdir(target_dic)
    new_filename = str(imagenet_class) + '_' + filename
    target = os.path.join(target_dic,new_filename)
    copyfile(source, target)

