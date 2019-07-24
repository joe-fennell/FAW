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

#TODO: next, training and validation image lists saved in training folder


def setup_training_run_folder():

    """Gets the test number from the user and checks to see if such a test
    already exits. Returns a `str` of length 4 representing the test number.

    Returns:
        int: Training run number.
        str: Path to training run save folder."""

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

    # Set up logging to file
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format(save_folder, number))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return number, save_folder


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
        none
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

 



