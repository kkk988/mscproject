"""
This script is used for training the Model 3 in our project.
The Model 1 is training the VGG16 network with random initisation weights.
Training data is the original ImageNet training set and a new class containing generated images.
"""

import os
import numpy as np
import imageio
from random import sample
import tensorflow.keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import matplotlib.pyplot as plt

# Set active GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set the paths to the training and validation images
train_dir = "/home/yuqi/new_class/new_train/train"
validation_dir = "/home/yuqi/new_class/new_train/val"

# Checkpoint to save model
checkpoint_filename = '/home/yuqi/new_class/checkpoint/saved_model.h5'
# History file to save the value at each epoch.
history_filename = '/home/yuqi/new_class/checkpoint/history_log.csv'

# Create a list of the training images, with the image path and class
train_sets = [(os.path.join(dp, f), dp.split('/')[-1]) for dp, dn, fn in os.walk(train_dir) for f in fn]
# Divides this into the paths and class labels
x_train, y_cls_train = zip(*train_sets)

# Change directory to the validation set
os.chdir(validation_dir)
list_of_dirs=os.listdir()
# Sort classes in ascending order so they will match the order that they exist
# in the prediction layer of VGG16 which is sorted alphanumerically
list_of_dirs.sort()
# Create a list of these 1001 classes
class_numbers = range(1001)
# Assign each class label this number (this will be the number in the prediction layer)
y_classes = {list_of_dirs[i]:class_numbers[i] for i in range(1001)}

# Get the appropriate class number for each image in the list
y_train = [y_classes[y] for y in y_cls_train]
# Create a one-hot vector of 1s and 0s to code the classes
y_train_oh = tensorflow.keras.utils.to_categorical(np.copy(y_train))


# Get the path and classes for the validation set
val_sets = [(os.path.join(dp, f), dp.split('/')[-1]) for dp, dn, fn in os.walk(validation_dir) for f in fn]
# Separate into validation path and class
x_val, y_cls_val = zip(*val_sets)

# Get the labels for the validation set
y_val = [y_classes[y] for y in y_cls_val]
# Create one hot vector for validation
y_val_oh = tensorflow.keras.utils.to_categorical(np.copy(y_val))

type(np.asarray([0, 1, 2]))


# Here is a keras sequence generator. Because all of the ImageNet files
# cannot fit into RAM, this creates a list of all files and samples from it
# in order, one batch at a time.
# This also preprocesses the data, taking a sample of 1000 images, and
# subtracting the mean of those images from the selected image,
# normalises to values between 0 and 1, and rescales image values
# by subtracting the SD

class ImagenetSequence(tensorflow.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size,
                 featurewise_normalization=None,
                 featurewise_normalization_sample_size=900):
        """
        featurewise_normalization ---
                a tuple (mean, std), "per_channel" or "global" -
                - if a tuple, center and standardize accordingly.
                per_channel calculates mean & std for a sample.
                "per_channel" processes RGB independently, "global" calculates single mean/std
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle_order = np.arange(len(self.x))
        np.random.shuffle(self.shuffle_order)
        self.x_shuf = self.x[self.shuffle_order]
        self.y_shuf = self.y[self.shuffle_order]

        if type(featurewise_normalization) is tuple:
            self.mean, self.std = featurewise_normalization
            if type(self.mean) is np.ndarray:
                self.featurewise_normalization = "per_channel"
            else:
                self.featurewise_normalization = "global"

        elif featurewise_normalization is not None:

            print(featurewise_normalization)
            self.featurewise_normalization = featurewise_normalization

            print(self.featurewise_normalization)
            self.featurewise_normalization_sample_size = featurewise_normalization_sample_size

            self.mean, self.std = self.compute_stats()
        else:
            self.featurewise_normalization = None
            self.mean = 0.
            self.std = 1.

    def compute_stats(self):
        """
        train_x_files --- a list of files, one per training example
        sample_size --- number of files to sample from train_x_files to calculate stats
        std_per_channel --- standardize and center independently for each RGB
        """

        if self.featurewise_normalization == "per_channel":
            ax = (0, 1, 2)  # Standardize & center for each channel independently

        elif self.featurewise_normalization == "global":
            ax = (0, 1, 2, 3)  # Overall standardize & center

        else:
            raise Exception("featurewise normalization should be 'per_channel' or 'global'")

        sample_files = sample(list(self.x), self.featurewise_normalization_sample_size)
        sample_matrix = np.asarray([imageio.imread(im) * 1. / 255 for im in sample_files], dtype='float32')
        mean = np.mean(sample_matrix, axis=ax)
        std = np.std(sample_matrix, axis=ax)
        return mean, std

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_shuf[idx * self.batch_size:(idx + 1) * self.batch_size]

        def read_image(self, file_name):
            image = imageio.imread(file_name)
            image = image * 1. / 255

            if self.featurewise_normalization is not None:
                image = (image - self.mean) / self.std

            return image

        return np.array([read_image(self, file_name) for file_name in batch_x]), np.array(batch_y, dtype='int')

    def on_epoch_end(self):
        np.random.shuffle(self.shuffle_order)


# Select batch size
batch_size = 32

# Create training sequence using above sequence generator
train_seq = ImagenetSequence(np.asarray(x_train),
                             np.asarray(y_train_oh),
                             batch_size=batch_size,
                             featurewise_normalization="global")

# Apply the exact same preprocessing done on the training images to the validation images
val_seq = ImagenetSequence(np.asarray(x_val),
                           np.asarray(y_val_oh),
                           batch_size=batch_size,
                           featurewise_normalization=(train_seq.mean, train_seq.std))
val_seq.__getitem__(0)

# Import VGG16 with random initisation weights
model = VGG16(weights=None,
	include_top=True,
	input_shape=(224, 224, 3), classes = 1001)
model.summary()

# Add dropout after the final two fully connected layers
model2 = models.Sequential()
for layer in model.layers:
	model2.add(layer)
	if layer.name in ['fc1','fc2']:
		model2.add(Dropout(.5))
	model2.summary()

# Compile the model
model2.compile(optimizer=optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1.0, decay=0.0),
		loss="categorical_crossentropy",
		metrics=["acc", "top_k_categorical_accuracy"])

# Save the best model after each epoch
checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# Create the csv that will log the training values
csv_logger = CSVLogger(history_filename, append=True, separator=';')

# The size of each training step within the training and validation epochs
step_size_train = len(train_sets) // batch_size
step_size_val = len(val_sets) // batch_size
print("train step size:", step_size_train)
print("train step val:", step_size_val)

epochs = 200

# Fit the model
history = model2.fit_generator(train_seq,
                        steps_per_epoch=step_size_train,
                        epochs=epochs,
                        validation_data=val_seq,
                        validation_steps=step_size_val,
                        callbacks=[csv_logger,checkpoint],
                        max_queue_size=16,
                        workers=8,
                        use_multiprocessing=True,
                        verbose=1,
                        )

history.params.items()

# Change the directory to save the history file
os.chdir('/home/yuqi/new_class/checkpoint/')

# Save the history file
import pickle
with open('train_new_from_scratch.pickle','wb') as f:
	 pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
top_k_acc = history.history['top_k_categorical_accuracy']
val_top_k_acc = history.history['val_top_k_categorical_accuracy']
epochs = range(1, len(acc) + 1)

# Plot figures to show the process of training
plt.plot(epochs, acc, 'o-', label='Training acc', color='cornflowerblue')
plt.plot(epochs, val_acc, 's-', label='Validation acc', color='coral')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('Training and validation accuracy.png', dpi=500)

plt.figure()

plt.plot(epochs, top_k_acc, 'o-', label='Training top_5_acc', color='cornflowerblue')
plt.plot(epochs, val_top_k_acc, 's-', label='Validation top_5_acc', color='coral')
plt.title('Training and validation top 5 accuracy')
plt.legend()
plt.savefig('Training and validation top 5 accuracy.png', dpi=500)

plt.figure()

plt.plot(epochs, loss, 'o-', label='Training loss', color='cornflowerblue')
plt.plot(epochs, val_loss, 's-', label='Validation loss', color='coral')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('Training and validation loss.png', dpi=500)
