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
                                save_model)


# Set TensorFlow config
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# Import config from config file
config_file = os.path.dirname(os.path.abspath(__file__)) + '/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)

# Initial setup
TRAINING_NUMBER, SAVE_DIR = setup_training_run_folder()


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



