"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

BuildClassifier.py

Script to train and build the Fall Armyworm Classifier.
"""

import tensorflow as tf
import keras
import pandas as pd
import pickle
import numpy as np
import glob
import pathlib
import json
import warnings
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import array_to_img
from classification_models.resnet import ResNet18
from skimage.segmentation import slic

# Set TensorFlow config
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
tf.logging.set_verbosity(tf.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# GLOBALS
PROJ_DIR = str(pathlib.Path(__file__).parents[1])
TRAIN_DIR = PROJ_DIR + '/data/train'
VALIDATION_DIR = PROJ_DIR + '/data/validation'
KMEANS_3 = pickle.load(open(PROJ_DIR + '/models/kmeans_224.sav', 'rb'))
BATCH_SIZE = 1
IMG_W, IMG_H = 224, 224
MLP_INPUT = (7, 7, 512)


def _predict(data, model, number_segments=2000):
    # returns label image
    # segment the image
    test_segments = slic(data,
                         n_segments=number_segments,
                         compactness=0.1,
                         sigma=0,
                         convert2lab=False)

    # calculate seg stats
    test_set = _calculate_segment_stats(data, test_segments)
    # predict
    test_set_segment_labels = model.predict(test_set)
    # code via broadcasting
    return test_set_segment_labels[test_segments]


def _calculate_segment_stats(data, segments):
    # turn the image into a 2D array (pix by channel)
    d1_flat = pd.DataFrame(np.ravel(data).reshape((-1, 3)))
    # add the label vector
    d1_flat['spID'] = np.ravel(segments)
    # calculate the mean by segment
    return d1_flat.groupby('spID').mean().values


def _preprocess(im):
    # predict labels for data via K means with 3 clusters
    im2 = np.array(im)
    im_labels = _predict(np.float64(im2 / 255), KMEANS_3)
    # imgarr = img_to_array(im, data_format=None)
    im2[:, :, 0][im_labels == 0] = 0
    im2[:, :, 1][im_labels == 0] = 0
    im2[:, :, 2][im_labels == 0] = 0
    return array_to_img(im / 255)


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
    mlp_json = mlp_model.to_json()
    with open(PROJ_DIR + '/models/MLP_CNN_model.json', 'w', encoding='utf-8') as f:
        json.dump(mlp_json, f)
    mlp_model.save_weights(PROJ_DIR + '/models/MLP_CNN_weights.h5')

    # combine ResNet18 base and trained mlp
    model = models.Model(inputs=base_model.input,
                         outputs=mlp_model(base_model.output))

    return model
