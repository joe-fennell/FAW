"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

FAW_detector.py

Script to identify Fall Armyworm from a given image..

Outputs true or false for Fall Armyworm detected.

Will generate and train the RESNET18 based classifier if it is not present.
"""

# TODO: investigate whether we can avoid hard coded image dimensions
# TODO: implement argparse for test number and recalculating of weights

# TODO: change the docstrings to NumPy format

import tensorflow as tf
import keras
import pandas as pd
import pickle
import numpy as np
import glob
import pathlib
import imagecheck.ImageCheck as ImageCheck
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import array_to_img
from classification_models.resnet import ResNet18
from skimage.segmentation import slic

# Set TensorFlow config
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# GLOBALS
BASE_PATH = str(pathlib.Path(__file__).parent)
MLP_JSON = BASE_PATH + '/models/MLP_CNN_model.json'
MLP_WEIGHTS = BASE_PATH + '/models/MLP_weights.h5'
TRAIN_DIR = '/mnt/data/train'
VALIDATION_DIR = '/mnt/data/validation'
RESNET_DIR = BASE_PATH + 'models/FC_Resnet.h5'
BATCH_SIZE = 1
IMG_W, IMG_H = 224, 224


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
    # predict labels for data via K means
    kmeans_3clusters = pickle.load(open(
        BASE_PATH + '/models/kmeans_224.sav', 'rb'))
    im2 = np.array(im)
    im_labels = _predict(np.float64(im2 / 255), kmeans_3clusters)
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

    """ Returns a DirectoryIterator yielding tuples of (x, y) where x is a numpy
    array containing a batch of images with shape (BATCH_SIZE, *target_size,
    channels) and y is a numpy array of corresponding labels.
    https://keras.io/preprocessing/image/#flow_from_directory """

    iterator = generator.flow_from_directory(data_dir, target_size=target_size,
                                             batch_size=BATCH_SIZE,
                                             class_mode=class_mode,
                                             shuffle=shuffle)

    return iterator


class FAW_classifier:

    # TODO: write docstring for class.

    def __init__(self,
                 classifier_json=MLP_JSON,
                 classifier_weights=MLP_WEIGHTS):

        self._classifier = self.load_classifier(classifier_weights,
                                                classifier_json)

    def load_classifier(mlp_weights_path, mlp_json_path):
        """ Loads classifier via saved weights.

        Args:
            weights path: Path to the saved CNN classifier weights.
        Returns:
            The loaded fully connected and trained ResNet18 model.
        """
        # TODO: finish
        # for training ResNet model if not present in dir.
        weights = pathlib.Path(mlp_weights_path)
        json_file = pathlib.Path(mlp_json_path)

        if not weights.is_file() or not json_file.is_file():
            # if we don't have the required files, regenerate the classifier.
            return _make_classifier()

        # If we already have the models weights
        with json_file.open() as f:
            loaded_model_json = f.read()
        loaded_mlp = model_from_json(loaded_model_json)
        loaded_mlp.load_weights(str(weights))

        # ResNet18 base
        base_model = ResNet18(input_shape=(IMG_W, IMG_H, 3),
                              weights='imagenet',
                              include_top=False)
        # Tack the loaded MLP on the the ResNet18 base and return
        model = models.Model(inputs=base_model.input,
                             outputs=loaded_mlp(base_model.output))
        return model

    def process_image(image_path):
        """ Processes an image using imagecheck.check_and_crop function.

        Args:
            image_path: Path to the image to be process.
        Returns:
            A cropped image containing the worm:
        Raises:
            ImageCheckError: Raised from imagecheck.py if the image does not
            meet the required standards or no worm is found.
        """
        return ImageCheck.check_and_crop(image_path)


def _make_classifier():

    """Make classifier if saved weights not present."""

    # Otherwise, train it from saved features data
    def get_num_samples(path, ext):
        """Finds the number of .jpg samples in a given dir."""
        find_str = path + '/**/*.{}'.format(ext)
        num_jpgs = len(glob.glob(find_str, recursive=True))
        return num_jpgs

    NB_TRAIN_SAMPLES = get_num_samples(TRAIN_DIR, '.jpg')
    NB_VALIDATION_SAMPLES = get_num_samples(VALIDATION_DIR, '.jpg')

    # Build iterators to access training and validation data
    datagen = ImageDataGenerator(rotation_range=90,
                                 preprocessing_function=_preprocess,
                                 fill_mode='nearest')

    train_iter = get_iterator(datagen, TRAIN_DIR)
    valid_iter = get_iterator(datagen, VALIDATION_DIR)

    # ResNet18 base
    base_model = ResNet18(input_shape=(IMG_W, IMG_H, 3),
                          weights='imagenet',
                          include_top=False)

    # get a numpy array of predictions from the train data
    train_data = base_model.predict_generator(train_iter,
                                              (NB_TRAIN_SAMPLES //
                                               BATCH_SIZE),
                                              verbose=1)
    # get a numpy array of predictions from the validation data
    validation_data = base_model.predict_generator(
        valid_iter,
        (NB_VALIDATION_SAMPLES
         // BATCH_SIZE),
        verbose=1)
    # MLP cap
    mlp_model = Sequential()
    mlp_model.add(Flatten(input_shape=train_data.shape[1:]))
    mlp_model.add(Dense(1024, activation='relu'))
    mlp_model.add(Dropout(0.5))
    mlp_model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(lr=0.0001, amsgrad=True)
    mlp_model.compile(optimizer=adam,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    # save training features
    # np.savez_compressed(BASE_PATH + '/models/bottleneck_features_train',
    #                     bottleneck_features_train)
    # np.savez_compressed(BASE_PATH + '/models/bottleneck_features_validation',
    #                     bottleneck_features_validation)

    # get the number of classes and their labels in original order
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
    with open(BASE_PATH + '/models/MLP_CNN_model.json', 'w') as f:
        f.write(mlp_json)
    mlp_model.save_weights(BASE_PATH + '/models/MLP_weights.h5')

    # combine ResNet18 base and trained mlp
    model = models.Model(inputs=base_model.input,
                         outputs=mlp_model(base_model.output))

    return model


_make_classifier()
