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
                            get_iterator, preprocess_images)
from FAW import ImageCheck


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

# Preprocess training data
print("\n\nImage preprocessing.\n\nBACKUP YOUR DATA BEFORE CONTINUINING")
print("Processed images are saved over their originals and images that fail"
      " are removed.")
input("Press any key to begin processing.")
preprocess_images(config['training_dir'], config['img_input_shape'])
preprocess_images(config['validation_dir'], config['img_input_shape'])




# Build iterators to access training and validation data
datagen = ImageDataGenerator(rotation_range=90,
                             fill_mode='nearest')

train_iter = get_iterator(datagen,
                          config['training_dir'],
                          config['img_input_shape'][:2],
                          config['batch_size'])
valid_iter = get_iterator(datagen,
                          config['validation_dir'],
                          config['img_input_shape'][:2],
                          config['batch_size'])

# ResNet18 base
base_model = ResNet18(input_shape=config['img_input_shape'],
                      weights='imagenet',
                      include_top=False)

# Calculate the FC weights
# get a numpy array of predictions from the train data
print("Running prediction generator for FC training.")
mlp_train_data = base_model.predict_generator(train_iter,
                                     (get_num_samples(config['training_dir'])
                                      // config['batch_size']))

# get a numpy array of predictions from the validation data
mlp_validation_data = base_model.predict_generator(
    valid_iter,
    (get_num_samples(config['validation_dir'])
    // config['batch_size']))

# get the number of classes and their labels in original order
datagen_top = ImageDataGenerator()
train_iter_top = get_iterator(datagen_top,
                              config['training_dir'],
                              config['img_input_shape'][:2],
                              class_mode='categorical')
valid_iter_top = get_iterator(datagen_top,
                              config['validation_dir'],
                              config['img_input_shape'][:2])
train_labels = train_iter_top.classes
validation_labels = valid_iter_top.classes
num_classes = len(train_iter_top.class_indices)

# MLP cap
mlp_model = Sequential()
mlp_model.add(Flatten(input_shape=mlp_train_data.shape[1:]))
mlp_model.add(Dense(1024, activation='relu'))
mlp_model.add(Dropout(rate=config['model_parameters']['mlp_dropout_rate']))
mlp_model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(lr=config['model_parameters']['mlp_optimizer_lr'],
                             amsgrad=True)
mlp_model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

print(len(mlp_train_data), len(train_labels))

# Calculate and save the pre-trained weights
history = mlp_model.fit(mlp_train_data, train_labels,
                    epochs=25,
                    batch_size=config['batch_size'],
                    validation_data=(mlp_validation_data, validation_labels))
model.save_weights(SAVE_DIR +
                   '/{}_bottleneck_fc_model_amsgrad.h5'.format(TRAINING_NUMBER))
history_dict = history.history
save = SAVE_DIR + '/{}_bottleneck_history_amsgrad.json'.format(TRAINING_NUMBER)
json.dump(history_dict, open(save, 'w'))


model = models.Model(inputs=base_model.input,
                     outputs=mlp_model(base_model.output))



