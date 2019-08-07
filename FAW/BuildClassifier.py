"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

BuildClassifier.py

Script to train and build the Fall Armyworm Classifier.
"""

# TODO: add code to supress Keras "Using backend ___" console output

import pickle
import glob
import pathlib
import warnings
import os
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import array_to_img
from classification_models.resnet import ResNet18
from skimage.segmentation import slic

# Set TensorFlow config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # silence tensorflow messages
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# GLOBALS
PROJ_DIR = str(pathlib.Path(__file__).parent)
TRAIN_DIR = PROJ_DIR + '/data/train'
VALIDATION_DIR = PROJ_DIR + '/data/validation'
KMEANS_3 = pickle.load(open(PROJ_DIR + '/models/kmeans_224.sav', 'rb'))
BATCH_SIZE = 1
IMG_W, IMG_H = 224, 224
MLP_INPUT = (7, 7, 512)


# TODO: remove anything to do with training model. This script should become
# purely a loading in of a saved model that has come from the training branch
# of the program


def get_iterator(generator,
                 data_dir,
                 target_size=(IMG_W, IMG_H),
                 batch_size=BATCH_SIZE,
                 class_mode=None,
                 shuffle=False):

    """Wrapper for Keras iterator from generator.

    Args:
        generator: Keras generator
        data_dir (str): path to the required data.
        target_size (tuple): tuple of ints containing image width and height.
        batch_size (int): contains batch size number.
        class_mode: see Keras docs below.
        shuffle (bool): True to shuffle order of data in dir.

    Returns:
        a DirectoryIterator yielding tuples of (x, y) where x is a numpy
        array containing a batch of images with shape (BATCH_SIZE,
        *target_size, channels) and y is a numpy array of corresponding labels.
        https://keras.io/preprocessing/image/#flow_from_directory """

    iterator = generator.flow_from_directory(data_dir, target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode=class_mode,
                                             shuffle=shuffle)

    return iterator


def make_classifier(weights_path=None):

    """Make classifier. Train it if weights not present.

    Args:
        weights_path (str): Path to the weights file.

    Returns:
        Keras model with ResNet18 base and MLP cap."""

    # ResNet18 base
    base_model = ResNet18(input_shape=(IMG_W, IMG_H, 3),
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

    if weights_path is not None:  # Check if weights file is valid.
        weights_path = pathlib.Path(weights_path)
        if weights_path.is_file() and weights_path.suffix == '.h5':
            mlp_model.load_weights(str(weights_path))
            model = models.Model(inputs=base_model.input,
                                 outputs=mlp_model(base_model.output))
            return model
        # Otherwise, invalid weights file. Print to console and train.
        print("ERROR: Weights file provided is invalid. Press any key to "
              "retrain model.")
        input()

    # Otherwise, train it from saved features data
    def get_num_samples(path, ext):
        """Finds the number of .jpg samples in a given dir."""
        find_str = path + '/**/*{}'.format(ext)
        num_jpgs = len(glob.glob(find_str, recursive=True))
        return num_jpgs

    NB_TRAIN_SAMPLES = get_num_samples(TRAIN_DIR, '.jpg')
    NB_VALIDATION_SAMPLES = get_num_samples(VALIDATION_DIR, '.jpg')

    if NB_TRAIN_SAMPLES == 0 or NB_VALIDATION_SAMPLES == 0:
        raise Exception("No training data in dir. Cannot make classifier.")

    # Build iterators to access training and validation data
    datagen = ImageDataGenerator(rotation_range=90,
                                 preprocessing_function=_preprocess,
                                 fill_mode='nearest')

    train_iter = get_iterator(datagen, TRAIN_DIR)
    valid_iter = get_iterator(datagen, VALIDATION_DIR)

    # get a numpy array of predictions from the train data
    train_data = base_model.predict_generator(train_iter,
                                              (NB_TRAIN_SAMPLES //
                                               BATCH_SIZE),
                                              verbose=1)
    # get a numpy array of predictions from the validation data
    validation_data = base_model.predict_generator(valid_iter,
                                                   (NB_VALIDATION_SAMPLES
                                                    // BATCH_SIZE),
                                                   verbose=1)

    datagen_top = ImageDataGenerator()
    train_iter_top = get_iterator(datagen_top, TRAIN_DIR,
                                  class_mode='categorical')
    valid_iter_top = get_iterator(datagen_top, VALIDATION_DIR)
    train_labels = train_iter_top.classes
    validation_labels = valid_iter_top.classes

    # Train the MLP that caps the ResNet model.
    mlp_model.fit(train_data, train_labels,
                  epochs=25,
                  batch_size=BATCH_SIZE,
                  validation_data=(validation_data, validation_labels))

    # Save MLP model to file
    mlp_model.save_weights(PROJ_DIR + '/models/MLP_CNN_weights.h5')

    # combine ResNet18 base and trained mlp
    model = models.Model(inputs=base_model.input,
                         outputs=mlp_model(base_model.output))

    return model
