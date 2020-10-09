"""
This script uses the visualisation toolkit (https://github.com/raghakot/keras-vis)
to produce images. To speed things up, I firstly ran it on each class
to find out how many epochs it took to create a top 5 image for each class
Then I saved this as a pickle file in the folder, and open this each time
to generate an image with the same number of epochs.
Obviously when running this script, it does not produce an image which is 
always top 5, so we come back later just to select the ones which make it.
After generating some images, we found that some classes are still difficult 
to produce, so I increased the number of epochs for them (see epochs variable below). 
This version of the script checks if enough images have been produced in each 
class (1000) and if it is less than that it tries to generate more images.
Whether the images are top 5 or not, and whether they are
deleted, is in the "check_top_5.py" script 
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")



# This part gets the mean and SD for each channel (RGB) for the class
# so that a noise image can be produced which is close to that classes
# typical colour.
def get_features(imagenet_class):
	imagenet_class = imagenet_class
	import os
	import numpy as np
	from keras.preprocessing.image import img_to_array  
	from skimage.io import imread 
	from skimage.transform import resize
	import time
	import random
	from PIL import Image
	from PIL import ImageStat

	imagenet_directory = "/fast-data20/datasets/ILSVRC/2012/clsloc/train"
	os.chdir(imagenet_directory)
	list_of_dirs = os.listdir()
	list_of_dirs.sort()
	imagenet_class = imagenet_class
	class_label = list_of_dirs[imagenet_class]
	random_image_dir = os.path.join(imagenet_directory, class_label)
	os.chdir(random_image_dir)
	list_of_files = os.listdir()
	amount_of_files = (len(list_of_files))
	random.seed(int(time.time()))

    # Randomly selete 100 images in the original ImageNet dataset for this class
	random_indices = random.sample(range(amount_of_files-1), 100)
	image_mean = np.zeros([100,3])
	image_std = np.zeros([100,3])
	for n in range(100):
		imagepath = list_of_files[random_indices[n]]
		Imagefile = Image.open(imagepath)
		image_stats = ImageStat.Stat(Imagefile)
		image_mean[n,:] = image_stats.mean
		image_std[n,:] = image_stats.stddev
	
	group_image_mean = np.mean(image_mean, axis=0)
	group_image_std = np.mean(image_std, axis=0)
	return group_image_mean, group_image_std


# This part creates the noise image using those features.
def create_random_array(imagenet_class):
	import numpy as np
	import time
	from keras import backend as K
	imagenet_class = imagenet_class
	mean,std = get_features(imagenet_class)
	shape = [224, 224]
	
	np.random.seed(int(time.time()))
	x0 = np.random.random(shape)
	# normalize around mean=0, std=1
	x0 = (x0 - np.mean(x0)) / (np.std(x0) + K.epsilon())
	# and then around the desired mean/std
	x0 = (x0 * std[0]) + mean[0]
	np.random.seed(int(time.time()))
	x1 = np.random.random(shape)
	# normalize around mean=0, std=1
	x1 = (x1 - np.mean(x1)) / (np.std(x1) + K.epsilon())
	# and then around the desired mean/std
	x1 = (x1 * std[1]) + mean[1]
	np.random.seed(int(time.time()))
	x2 = np.random.random(shape)
	# normalize around mean=0, std=1
	x2 = (x2 - np.mean(x2)) / (np.std(x2) + K.epsilon())
	# and then around the desired mean/std
	x2 = (x2 * std[2]) + mean[2]
	random_image = np.zeros([224, 224, 3])
	random_image[:,:,0] = x0[:,:]
	random_image[:,:,1] = x1[:,:]
	random_image[:,:,2] = x2[:,:]
	return random_image
                                       
# This part actually generates the images
def make_image(imagenet_class,sample):
	imagenet_class = imagenet_class
	sample = sample
	import os 
	import numpy as np
	# this is a record of how many top 5 images there are in each class so far
	# note the classes are in order that they appear in the predictions layer
	# in vgg16, which is in alphabetical order
	top_5_complete = np.load('/home/yuqi/generate_images/top_5_by_class.npy')
	top_5_value = top_5_complete[imagenet_class]
	if top_5_value < 1000:
		from keras.applications import VGG16
		from keras import activations
		from keras.applications.vgg16 import preprocess_input
		from keras.applications.vgg16 import decode_predictions
		import time
		from matplotlib import pyplot as plt
		import numpy as np
		from vis.utils import utils
		from vis.visualization import visualize_activation
		from vis.input_modifiers import Jitter
		from PIL import Image
		# Build the VGG16 network with ImageNet weights
		model = VGG16(weights='imagenet', include_top=True)
		layer_idx = utils.find_layer_idx(model, 'predictions')
		model.layers[layer_idx].activation = activations.linear
		model = utils.apply_modifications(model)
		# shape=224,224,3
		import pickle
		import os
		pickle_base_dir = '/home/yuqi/generate_images/pickle'
		pickle_class = str(imagenet_class)
		pickle_extension = '.pickle'
		pickle_title = pickle_class + pickle_extension
		pickle_file = os.path.join(pickle_base_dir, pickle_title)
		infile = open(pickle_file,'rb')
		[epoch, accuracy] = pickle.load(infile)
		infile.close()
		######################################
		# This is where in increase the number of epochs to help the generation
		# of a top 5 image.
		epoch = epoch
		epoch = int(epoch)
		start_sample = sample * 25
		stop_sample = start_sample + 25
		for iteration in range(start_sample, stop_sample):
			start_time = time.time()
			random_seed_input = create_random_array(imagenet_class)
			img = visualize_activation(model, layer_idx, filter_indices=imagenet_class, seed_input=random_seed_input, max_iter=epoch, input_modifiers=[Jitter(16)], verbose=True)
			image_prefix = 'image'
			image_class = str(imagenet_class)
			image_sample = str(iteration)
			image_extension = '.jpeg'
			image_space = '_'
			image_title_jpeg = image_prefix+image_space+image_sample+image_extension
			image_base_dir = '/home/yuqi/generate_images/generated_images_colour'
			image_dir = os.path.join(image_base_dir, image_class)
			if not os.path.exists(image_dir):
				os.mkdir(image_dir)
			os.chdir(image_dir)
			new_img = Image.fromarray(img)
			new_img.save(image_title_jpeg)
			elapsed_time = time.time() - start_time
			print('time', elapsed_time)

import multiprocessing
for sample in range(0,15):
	for imagenet_class in range(1000):
		process_train = multiprocessing.Process(target=make_image, args=(imagenet_class,sample))
		process_train.start()
		process_train.join()
