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

import pathlib
import imagecheck.ImageCheck as ImageCheck
import models.BuildClassifier as BC
from keras import models
from keras.models import model_from_json
from classification_models.resnet import ResNet18


# GLOBALS
BASE_PATH = str(pathlib.Path(__file__).parent)
MLP_JSON = BASE_PATH + '/models/MLP_CNN_model.json'
MLP_WEIGHTS = BASE_PATH + '/models/MLP_CNN_weights.h5'
IMG_W, IMG_H = 224, 224


class FAW_classifier:
    """Classifer that loads CNN and crops and classifies fed images."""

    def __init__(self,
                 classifier_json=MLP_JSON,
                 classifier_weights=MLP_WEIGHTS):
        """
        Args:
            classifier_json (str): path to the .json file containing the MLP
            structure.
            classifier_weights (str): path to the .h5 MLP weights file.
        """

        self._classifier = self.load_classifier(classifier_weights,
                                                classifier_json)

    def load_classifier(self, mlp_weights_path, mlp_json_path):
        """ Loads classifier via saved weights.

        Args:
            weights_path (str): Path to the saved CNN classifier weights.
        Returns:
            The loaded fully connected and trained ResNet18 model.
        """
        # TODO: finish
        # for training ResNet model if not present in dir.
        weights = pathlib.Path(mlp_weights_path)
        json_file = pathlib.Path(mlp_json_path)
        print(mlp_json_path)
        print(mlp_weights_path)

        if not weights.is_file() or not json_file.is_file():
            # if we don't have the required files, regenerate the classifier.
            return BC.make_classifier()

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

    def process_image(self, image_path):
        """Processes an image using imagecheck.check_and_crop function.

        Args:
            image_path (str): Path to the image to be process.
        Returns:
            A cropped image containing the worm.
        Raises:
            ImageCheckError: Raised from `imagecheck/ImageCheck.py`
            if the image does not meet the required standards or no
            worm is found.
        """
        return ImageCheck.check_and_crop(image_path)

    def predict(self, image_path):
        """Predict whether an image contains a Fall Armyworm.

        Args:
            image_path (str): Path to the image containing object to be
            classified.

        Returns:
            True if Fall Armyworm detected, else False.
        """
        image = self.process_image(image_path)

        return self._classifier.predict(image)


classifier = FAW_classifier(MLP_WEIGHTS, MLP_JSON)

data_dirs = ['train/', 'validation/']
categories = ['faw/', 'notfaw/']

for data_dir in data_dirs:
    for category in categories:
        loc_str = '/mnt/data/' + data_dir + category
        path = pathlib.Path(loc_str)
        imgs = list(path.glob('*.jpg'))
        for img in imgs:
            print(img)
            print(classifier.predict(str(img)))
