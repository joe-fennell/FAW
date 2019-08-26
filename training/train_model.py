"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

train_model.py

Script to train the Fall Armyworm Classifier.
"""
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications.resnet50 import ResNet50
from training_utils import (setup_training_run_folder, get_num_samples,
                            load_config, store_training_validation_file_list,
                            get_iterator, preprocess_images,
                            save_mlp_trained_model, save_cnn_trained_model)

# Set TensorFlow config
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=tf_config))


# Get config
config = load_config()

# Initial setup
TRAINING_NUMBER, SAVE_DIR, logger = setup_training_run_folder()

# Save the training data file lists to the save dir
store_training_validation_file_list((config['training_dir'],
                                     config['validation_dir']),
                                    SAVE_DIR,
                                    TRAINING_NUMBER,
                                    logger)

# Preprocess training data
logger.warning("Image replacement: %s",
               "\n\nImage preprocessing.\n\nBACKUP YOUR DATA BEFORE"
               " CONTINUINING")
logger.warning("Image replacement: %s",
               "Processed images are saved over their originals"
               " and images that fail are removed.")
input("Press any key to begin processing.")
preprocess_images(config['training_dir'], config['img_input_shape'], logger)
preprocess_images(config['validation_dir'], config['img_input_shape'], logger)


# Build iterators to access training and validation data
datagen = ImageDataGenerator(rotation_range=90,
                             fill_mode='nearest')

# class indices order required to be explicit in order to ensure notfaw label
# is 0 and faw label is 1
class_indices = ['notfaw', 'faw']
train_iter = get_iterator(datagen,
                          config['training_dir'],
                          config['img_input_shape'][:2],
                          config['batch_size'],
                          class_indices=class_indices)
valid_iter = get_iterator(datagen,
                          config['validation_dir'],
                          config['img_input_shape'][:2],
                          config['batch_size'],
                          class_indices=class_indices)

# ResNet18 base
base_model = ResNet50(input_shape=config['img_input_shape'],
                      weights='imagenet',
                      include_top=False)

# Calculate the FC weights
# get a numpy array of predictions from the train data
logger.info("Running prediction generator for FC training.")
mlp_train_data = base_model.predict_generator(train_iter,
                                              (get_num_samples(
                                               config['training_dir'])
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
                              class_mode='binary',  # binary, either FAW or not
                              class_indices=class_indices)

valid_iter_top = get_iterator(datagen_top,
                              config['validation_dir'],
                              config['img_input_shape'][:2],
                              class_indices=class_indices)
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

# Calculate and save the pre-trained weights
mlp_history = mlp_model.fit(
    mlp_train_data,
    train_labels,
    epochs=config['model_parameters']['mlp_num_training_epochs'],
    batch_size=config['batch_size'],
    validation_data=(mlp_validation_data, validation_labels))
save_mlp_trained_model(mlp_model, mlp_history, SAVE_DIR, TRAINING_NUMBER)

# Set num of CNN trainable layers and build the fully connected MLP
# and CNN model
num_trainable_layers = config['model_parameters']['cnn_num_trainable_layers']
for layer in base_model.layers[:-num_trainable_layers]:
    layer.trainable = False
model = models.Model(inputs=base_model.input,
                     outputs=mlp_model(base_model.output))

cnn_lr = config['model_parameters']['cnn_learning_rate']
adam = keras.optimizers.Adam(lr=cnn_lr, amsgrad=True)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])

logger.info("Model compiled")

logger.info(model.summary())

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rotation_range=90,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator()

train_iterator = get_iterator(train_datagen,
                              config['training_dir'],
                              config['img_input_shape'][:2],
                              class_mode='binary')
validation_iterator = get_iterator(validation_datagen,
                                   config['validation_dir'],
                                   config['img_input_shape'][:2],
                                   class_mode='binary')

# Fine-tune the ResNet model
finetune_history = model.fit_generator(
    train_iterator,
    steps_per_epoch=(get_num_samples(config['training_dir'])
                     // config['batch_size']),
    epochs=config['model_parameters']['cnn_num_training_epochs'],
    validation_data=validation_iterator,
    validation_steps=(get_num_samples(config['training_dir'])
                      // config['batch_size']))
save_cnn_trained_model(model, finetune_history, SAVE_DIR, TRAINING_NUMBER)

logger.info('FINISHED: %s', "Model training complete.")
