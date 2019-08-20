"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

ClassifierTools.py

Utility script for the Fall Armyworm Classifier.
"""

import glob
import os
import json
import pathlib
from keras.models import model_from_json 

def load_classifier(model_dir):

    """Make classifier.

    Args:
        model_dir (str): path to where the model is stored.


    Returns:
        keras.engine.training.Model: Loaded in Keras model from model dir.
    """

    try:
        model_structure_json = glob.glob(model_dir + '*_cnn_model.json')[0]
    except IndexError:
        raise FileNotFoundError('No saved model .json structure file present.')
    try:
        model_weights_h5 = glob.glob(model_dir + '*_cnn_weights.h5')[0]
    except IndexError:
        raise FileNotFoundError('No saved model .h5 weights file present.')

    with open(model_structure_json, 'r') as json_f:
        loaded_json = json_f.read()
        model = model_from_json(loaded_json)

    model.load_weights(model_weights_h5)

    return model

def load_config():
    """Loads the config.json file into memory.

    Returns:
        dict: Contains all config settings.
    """
    config_path = str(pathlib.Path(__file__).parents[1]) + '/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    return config
