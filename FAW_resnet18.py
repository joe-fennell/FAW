"""FAW CNN Classifier based on ResNet18"""

# TODO: investigate different image segmentation / preprocessing options
# TODO: investigate whether we can avoid hard coded image dimensions
# TODO: investigate NOTE on model setup in last section. Print layer.trainable
# for all to inspect whether this may be true.
# TODO: update keras version within simg


import tensorflow as tf
import keras
import pandas as pd
import pickle
import numpy as np
import json
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

##############################################################################
#                      Define useful variables                               #
##############################################################################

# load in k-means image segmentation made in jupyter notebook
kmeans_3clusters = pickle.load(open('/mnt/kmeans_224.sav', 'rb'))

# data dirs
train_dir = '/mnt/data/train'
validation_dir = '/mnt/data/validation'

batch_size = 1
img_width, img_height = 224, 224

nb_train_samples = 1130
nb_validation_samples = 280


def predict(data, model, number_segments=2000):
    """ returns label image"""
    # segment the image
    test_segments = slic(data,
                         n_segments=number_segments,
                         compactness=0.1,
                         sigma=0,
                         convert2lab=False)

    # calculate seg stats
    test_set = calculate_segment_stats(data, test_segments)
    # predict
    test_set_segment_labels = model.predict(test_set)
    # code via broadcasting
    return test_set_segment_labels[test_segments]


def calculate_segment_stats(data, segments):
    # turn the image into a 2D array (pix by channel)
    d1_flat = pd.DataFrame(np.ravel(data).reshape((-1, 3)))
    # add the label vector
    d1_flat['spID'] = np.ravel(segments)
    # calculate the mean by segment
    return d1_flat.groupby('spID').mean().values


def preprocess(im):
    im2 = np.array(im)
    im_labels = predict(np.float64(im2 / 255), kmeans_3clusters)
    # imgarr = img_to_array(im, data_format=None)
    im2[:, :, 0][im_labels == 0] = 0
    im2[:, :, 1][im_labels == 0] = 0
    im2[:, :, 2][im_labels == 0] = 0
    return array_to_img(im / 255)


##############################################################################
#                  Train FC network using bottleneck features                #
##############################################################################

def get_iterator(generator,
                 data_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False):

    """ Returns a DirectoryIterator yielding tuples of (x, y) where x is a numpy
    array containing a batch of images with shape (batch_size, *target_size,
    channels) and y is a numpy array of corresponding labels.
    https://keras.io/preprocessing/image/#flow_from_directory """

    iterator = generator.flow_from_directory(data_dir, target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode=class_mode,
                                             shuffle=shuffle)

    return iterator


recalculate = input("Recalculate FC weights? (y/n)")

if recalculate == 'y':

    # Build iterators to access training and validation data
    datagen = ImageDataGenerator(rotation_range=90,
                                 preprocessing_function=preprocess,
                                 fill_mode='nearest')

    train_iter = get_iterator(datagen, train_dir)
    valid_iter = get_iterator(datagen, validation_dir)

    model = ResNet18(input_shape=(img_width, img_height, 3),
                     weights='imagenet',
                     include_top=False)

    # get a numpy array of predictions from the train data
    bottleneck_features_train = model.predict_generator(train_iter,
                                                        (nb_train_samples //
                                                         batch_size))
    # NOTE: remove these if not needed
    # np.save('/mnt/saves/bottleneck_features_train_amsgrad.npy',
    #        bottleneck_features_train)

    # get a numpy array of predictions from the validation data
    bottleneck_features_validation = model.predict_generator(
        valid_iter,
        (nb_validation_samples
         // batch_size))
    # NOTE: remove these if not needed
    # np.save('/mnt/saves/bottleneck_features_validation_amsgrad.npy',
    #        bottleneck_features_validation)

    # get the number of classes and their labels in original order
    datagen_top = ImageDataGenerator()
    train_iter_top = get_iterator(datagen_top, train_dir,
                                  class_mode='categorical')
    valid_iter_top = get_iterator(datagen_top, validation_dir)
    train_labels = train_iter_top.classes
    validation_labels = valid_iter_top.classes
    num_classes = len(train_iter_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = bottleneck_features_train
    validation_data = bottleneck_features_validation

    # nb_validation_samples = len(generator_top.filenames)
    # nb_train_samples = len(generator_top.filenames)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    adam = keras.optimizers.Adam(lr=0.0001, amsgrad=True)
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=25,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))
    model.save_weights('/mnt/saves/bottleneck_fc_model_amsgrad.h5')
    history_dict = history.history
    json.dump(history_dict, open("/mnt/saves/bottleneck_history_amsgrad.json",
                                 'w'))

##############################################################################
#                              FineTune ResNet18                             #
##############################################################################

# build model
base_model = ResNet18(input_shape=(img_width, img_height, 3),
                      weights='imagenet', include_top=False)

for layer in base_model.layers[:-4]:
    layer.trainable = False

# Create a model
fullyconnected_model = Sequential()
fullyconnected_model.add(Flatten(input_shape=base_model.output_shape[1:]))
fullyconnected_model.add(Dense(1024, activation='relu'))
fullyconnected_model.add(Dropout(0.5))
fullyconnected_model.add(Dense(1, activation='sigmoid'))

fullyconnected_model.load_weights('/mnt/saves/bottleneck_fc_model_amsgrad.h5')

model = models.Model(inputs=base_model.input,
                     outputs=fullyconnected_model(base_model.output))

# NOTE: Should this trainable modifier not be just for the base model?
# Otherwise it may make only the top two layers of the 'model' trainable, as
# opposed to the top two layers of the 'base_model' trainable.

for layer in model.layers:
    print(layer, layer.trainable)

adam = keras.optimizers.Adam(lr=0.00001, amsgrad=True)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('model compiled')

model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rotation_range=90,
                                   preprocessing_function=preprocess,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess)

train_iterator = get_iterator(train_datagen, train_dir, class_mode='binary')

validation_iterator = get_iterator(validation_datagen,
                                   validation_dir, class_mode='binary')

# fine-tune the model
history = model.fit_generator(train_iterator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=100,
                              validation_data=validation_iterator,
                              validation_steps=(nb_validation_samples //
                                                batch_size))

model.save_weights('/mnt/saves/resnet18_fintunning_1_model_adadelta.h5')
history_dict = history.history
json.dump(history_dict,
          open("/mnt/saves/finetunning_history_amsgrad_amsgrad_lr00001.json",
               'w'))
print('model fit complete')
