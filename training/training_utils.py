"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

training_utils.py

Holds utility functions for model training.
"""
import glob
import shutil
import os
import sys
import logging
import json
import pathlib
import pickle
import cv2
from FAW import ImageCheck


def setup_training_run_folder():

    """Gets the test number from the user and checks to see if such a test
    already exits. Returns a `str` of length 4 representing the test number.

    Returns:
        int: Training run number.
        str: Path to training run save folder.
        logging.Logger: dedicated logger used to record script messages to
        log file and stdout.
    """

    number = input("Please enter the training run number (eg. 1, 2, 3): ")

    while len(number) < 4:
        number = '0' + number  # uniform length test number XXXX

    saves = list(glob.glob('saves/*'))

    for save in saves:
        if number in save:  # avoid learning rate / test number clash
            print("WARNING:")
            print("Training run number {} already exists.".format(number))
            answer = input("Are you sure you want to delete it? (y/n): ")
            if answer.lower() == 'y':
                shutil.rmtree('saves/{}'.format(number))
            else:
                raise ValueError("Training run number already exists in save files.")

    save_folder = 'saves/{}'.format(number)
    os.mkdir(save_folder)

    # copy the current config file over for posterity
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'config.json')
    new_path = os.path.join(save_folder, '{}_config.json'.format(number))
    shutil.copyfile(config_path, new_path)

    # Set up loggers to write to file
    rootLogger = logging.getLogger()  # logger used by other modules
    rootLogger.setLevel(20)
    fawLogger = logging.Logger('FAW_logger')  # dedicated FAW logger
    fawLogger.setLevel(10)

    streamFormatter = logging.Formatter(
        "%(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(streamFormatter)
    fawLogger.addHandler(streamHandler)
    
    logFormatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler("{0}/{1}.log".format(save_folder, number))
    fileHandler.setFormatter(logFormatter)
    fawLogger.addHandler(fileHandler)
    rootLogger.addHandler(fileHandler)

    return number, save_folder, fawLogger


def preprocess_images(images_dir, image_dims, logger):
    """Preprocesses the training data and saves a list of processed files to
    stop multiple processings of the same file.

    Args:
        images_dir (str): path to the images directory
        image_dims (tuple): (width, height, depth) of images
        logger (logging.Logger): logger object used to log messages
    Returns:
        None
    """
    find_str = images_dir + '/**/*.jpg'
    images = glob.glob(find_str, recursive=True)
    num_samples = get_num_samples(images_dir)

    # Load in the already processed file list
    proc_list_path = images_dir + '/processed_list.txt'
    if os.path.isfile(proc_list_path):
        with open(proc_list_path) as f:
            proc_list = f.read().split('\n')
    else:
        proc_list = []
    
    i = 1
    for image in images:
        image_name = image.split('/')[-1]
        if image not in proc_list:
            logger.info("Processing %s", " {} - {}/{}".format(
                image_name, i, num_samples))
            try:
                processed_image = ImageCheck.check_and_crop(image)
            except (ImageCheck.ObjectMissingError,
                    ImageCheck.WormMissingError,
                    ImageCheck.MultipleWormsError,
                    ImageCheck.TooBlurryError) as e:
                logger.info("Processing Error: %s",
                             "Image at: \n{} \n Produced error: {} \n Removing"
                             " image".format(image, e))
                os.remove(image)
                i = i + 1
                continue
            cv2.imwrite(image, processed_image)
            with open(proc_list_path, 'a') as f:
                f.write(image + '\n')
        else:
            logger.info("Skipping %s", " {} (already processed) - {}/{}".format(
                image_name, i, num_samples))
        i = i + 1

    
def get_num_samples(path):
    """Finds the number of .jpg samples in a given dir.

    Args:
        path (str): Path to the dir to find number jpgs.

    Returns:
        int: Number of jpgs found in the given dir.
        list: list containing str paths of all images in given dir."""


    find_str = path + '/**/*.jpg'
    num_jpgs = len(glob.glob(find_str, recursive=True))

    return num_jpgs


def save_model(model, test_number):
    """Saves the model to a saves folder in the saves directory as set in the
    config.

    Args:
        model (keras.engine.training.Model): model to be saved
        test_number (int): Unique test number used for the save folder name.

    Returns:
        None
    """

    model_json = model.to_json()

    with open("model_structure.json", "w") as f:
        f.write(model_json)


def load_config():
    """Loads the config data from the config file.

    Returns:
        dict
    """
    config_file = os.path.dirname(os.path.abspath(__file__)) + '/config.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    return config


def store_training_validation_file_list(data_paths, save_dir, train_num,
                                        logger):
    """Saves complete lists of the training and validation data in to the
    training dir.

    NOTE: Only detects images with .jpg extension.

    Args:
        data_paths (tuple): contains (training_data_path, validation_data_path)
        save_dir (str): path to the dir where file lists will be saved
        train_num (int): training run number
        logger (logging.Logger): logger object used to log messages

    Returns:
        None
    """
    training_dir = data_paths[0]
    validation_dir = data_paths[1]

    save_list = os.path.join(save_dir, '{}_train_valid_file_list.txt'.format(
        train_num))


    with open(save_list, "w") as f:

        def get_images(path):

            sub_dirs = [x[0] for x in os.walk(path)]
            sub_dirs.sort()

            for sub_dir in sub_dirs:
                images = glob.glob(sub_dir + '/*.jpg')
                
                # for dirs containing jpgs, write the dir path and files to save_list
                if len(images) > 0:
                    f.write(sub_dir + "\n")
                    for image in images:
                        f.write("     " + pathlib.Path(image).name + "\n")

        f.write("LIST OF FILES USED IN RUN {}\n".format(train_num))
        f.write("===============================\n")

        f.write("TRAINING\n")
        f.write("--------\n")

        get_images(training_dir)

        f.write("VALIDATION\n")
        f.write("----------\n")

        get_images(validation_dir)

    logger.info("File Generation: %s",
               "Training and validation files list generated.")


def get_iterator(generator,
                 data_dir,
                 target_size,
                 batch_size=1,
                 class_mode=None,
                 shuffle=False):

    """Returns a DirectoryIterator yielding tuples of (x, y) where x is a numpy
    array containing a batch of images with shape (batch_size, *target_size,
    channels) and y is a numpy array of corresponding labels.
    https://keras.io/preprocessing/image/#flow_from_directory
    
    Args:
        generator (keras.preprocessing.image.ImageDataGenerator): generator
            object from the keras module
        data_dir (str): path to the data directory
        target_size (tuple): (image_width (int), image_height (int))
        batch_size (int): size of the batches of data
        class_mode (str, optional): see above keras docs
        shuffle (bool, optional): whether to shuffle the data
        
    Returns:
        DirectoryIterator: A DirectoryIterator yielding tuples of (x, y) where
        x is a numpy array containing a batch of images with shape 
        (batch_size,
        *target_size,
        channels)
        and y is a numpy array of corresponding labels (from above docs)."""


    iterator = generator.flow_from_directory(data_dir, target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode=class_mode,
                                             shuffle=shuffle)

    return iterator


def save_mlp_trained_model(model, training_history, save_dir, train_num):
    """Saved the weights of the trained model and the training history.

    Args:
        model (keras.engine.training.Model): model to be saved
        training_history (keras.callbacks.History): training history of the
        model
        save_dir (str): address of dir to save the model in
        train_num (int): training number for naming the save files

    Returns:
        None
    """
    model.save_weights(save_dir +
                       '/{}_bottleneck_fc_model.h5'.format(train_num))
    history_dict = training_history.history
    save = save_dir + '/{}_bottleneck_history.json'.format(train_num)
    json.dump(history_dict, open(save, 'w'))


def save_cnn_trained_model(model, training_history, save_dir, train_num):
    """Saved the weights of the trained model and the training history.

    Args:
        model (keras.engine.training.Model): model to be saved
        training_history (keras.callbacks.History): training history of the
        model
        save_dir (str): address of dir to save the model in
        train_num (int): training number for naming the save files

    Returns:
        None
    """
    # Save the model training history
    history_dict = training_history.history
    history_save = save_dir + '/{}_cnn_history.json'.format(train_num)
    json.dump(history_dict, open(history_save, 'w'))
    # Save the model weights
    weights_save = save_dir + '/{}_cnn_weights.h5'.format(train_num)
    model.save_weights(weights_save)
    # Save the model structure to json file
    model_save = save_dir + '/{}_cnn_model.json'.format(train_num)
    model_json = model.to_json()
    json.dump(model_json, open(model_save, 'w'))

