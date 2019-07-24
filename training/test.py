import tensorflow as tf
import keras
import pandas as pd
import pickle
import numpy as np
import json
import glob
import os
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import array_to_img
from classification_models.resnet import ResNet18
from skimage.segmentation import slic
from training_utils import (setup_training_run_folder, get_num_samples,
                            save_model, load_config,
                            store_training_validation_file_list,
                            get_iterator)


# Set TensorFlow config
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=tf_config))


# Get config
config = load_config()

# Initial setup
TRAINING_NUMBER, SAVE_DIR = setup_training_run_folder()

# Save the training data file lists to the save dir
store_training_validation_file_list((config['training_dir'],
                                     config['validation_dir']),
                                    SAVE_DIR,
                                    TRAINING_NUMBER)


# TODO: continue from here:
#   make sure all variables from copy paste now come from config
#   change generator preprocess to ImageCheck.check_and_crop
#   will need to standardise output image to certain dimensions
#   this make require resizing images in check_and_crop to dimensions
#   that are passed in to the function when it is called

# Build iterators to access training and validation data
datagen = ImageDataGenerator(rotation_range=90,
                             preprocessing_function=preprocess,
                             fill_mode='nearest')

train_iter = get_iterator(datagen, train_dir)
valid_iter = get_iterator(datagen, validation_dir)

# ResNet18 base
base_model = ResNet18(input_shape=(224, 224, 3),
                      weights='imagenet',
                      include_top=False)

# MLP cap
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=MLP_INPUT))
mlp_model.add(Dense(1024, activation='relu'))
mlp_model.add(Dropout(rate=0.5))
mlp_model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(lr=0.0001, amsgrad=True)
mlp_model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model = models.Model(inputs=base_model.input,
                     outputs=mlp_model(base_model.output))



