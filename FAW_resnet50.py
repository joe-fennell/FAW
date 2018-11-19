import keras
from keras import applications
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from skimage.segmentation import slic
import pandas as pd
import pickle
import numpy as np
from keras.preprocessing.image import array_to_img
import json


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
    im_labels = predict(im2 / 255, kmeans_3clusters)
    # imgarr = img_to_array(im, data_format=None)
    im2[:, :, 0][im_labels == 0] = 0
    im2[:, :, 1][im_labels == 0] = 0
    im2[:, :, 2][im_labels == 0] = 0
    return array_to_img(im / 255)



##############################################################################
#                      Define useful variables                               #
##############################################################################

kmeans_3clusters = pickle.load(open('/mnt/kmeans_224.sav', 'rb'))

train_dir = 'data/train'
validation_dir = 'data/validation'

batch_size = 10
img_width, img_height = 224, 224

nb_train_samples = 1130
nb_validation_samples = 280


##############################################################################
#                  Train FC network using bottleneck features                #
##############################################################################

datagen = ImageDataGenerator(rotation_range=360,
                             preprocessing_function=preprocess,
                             zoom_range=0.2,
                             fill_mode='nearest')

model = applications.resnet50.ResNet50(input_shape=(img_width, img_height, 3),
                                       include_top=False, weights='imagenet')
generator = datagen.flow_from_directory('data/train',
                                        target_size=(img_width, img_height),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
bottleneck_features_train = model.predict_generator(generator,
                                                    nb_train_samples // batch_size)
# save the output as a Numpy array
np.save('bottleneck_features_train_resnet50.npy', bottleneck_features_train)


generator = datagen.flow_from_directory('data/validation',
                                        target_size=(img_width, img_height),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
bottleneck_features_validation = model.predict_generator(generator,
                                                         nb_validation_samples//batch_size)
np.save('bottleneck_features_validation_resnet50.npy', bottleneck_features_validation)


datagen_top = ImageDataGenerator()
generator_top = datagen_top.flow_from_directory(train_dir,
                                                target_size=(img_width,
                                                             img_height),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)

# nb_train_samples = len(generator_top.filenames)
num_classes = len(generator_top.class_indices)

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_train_resnet50.npy')

# get the class lebels for the training data, in the original order
train_labels = generator_top.classes

generator_top = datagen_top.flow_from_directory(validation_dir,
                                                target_size=(img_width,
                                                             img_height),
                                                batch_size=batch_size,
                                                class_mode=None,
                                                shuffle=False)

# nb_validation_samples = len(generator_top.filenames)

validation_data = np.load('bottleneck_features_validation_resnet50.npy')

validation_labels = generator_top.classes

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels,
                    epochs=150,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))
model.save_weights('/mnt/bottleneck_fc_model_resnet50.h5')
history_dict = history.history
json.dump(history_dict, open("/mnt/bottleneck_history_resnet50.json", 'w'))


##############################################################################
#                              FineTune ResNet18                             #
##############################################################################


# build model
base_model = applications.resnet50.ResNet50(input_shape=(img_width, img_height, 3),
                                       include_top=False, weights='imagenet')

# Create a model
fullyconnected_model = Sequential()
fullyconnected_model.add(Flatten(input_shape=base_model.output_shape[1:]))
fullyconnected_model.add(Dense(256, activation='relu'))
fullyconnected_model.add(Dropout(0.5))
fullyconnected_model.add(Dense(1, activation='sigmoid'))

fullyconnected_model.load_weights('bottleneck_fc_model.h5')

model = models.Model(inputs=base_model.input,
                     outputs=fullyconnected_model(base_model.output))

for layer in model.layers[:-3]:
    layer.trainable = False

rmsprop = keras.optimizers.RMSprop(lr=0.0001)
model.compile(optimizer=rmsprop,
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('model compiled')

model.summary()

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rotation_range=360,
                                   preprocessing_function=preprocess,
                                   zoom_range=0.2,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height,
                                                                 img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(img_height,
                                                                           img_width),
                                                              batch_size=batch_size,
                                                              class_mode='binary')

# fine-tune the model
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=20,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples // batch_size)

model.save_weights('/mnt/resnet50_fintunning_1_model.h5')
history_dict = history.history
json.dump(history_dict, open("/mnt/finetunning_history_resnet50.json", 'w'))
print('model fit complete')

