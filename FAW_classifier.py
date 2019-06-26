"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

FAW_detector.py

Script to identify Fall Armyworm from a given image..

Outputs true or false for Fall Armyworm detected.

Will generate and train the ResNet18 based classifier if it is not present.
"""

# TODO: investigate whether we can avoid hard coded image dimensions
# TODO: add argparse classification threshold option

import pathlib
import argparse
import cv2
import numpy as np
from FAW import ImageCheck
from FAW import BuildClassifier as BC


# GLOBALS
BASE_PATH = str(pathlib.Path(__file__).parent)
MLP_WEIGHTS = BASE_PATH + '/models/MLP_CNN_weights.h5'
IMG_DIMS = (224, 224)


class FAW_classifier:
    """Classifer that loads CNN and crops and classifies fed images."""

    def __init__(self, classifier_weights=MLP_WEIGHTS):
        """
        Args:
            classifier_weights (str): path to the .h5 MLP weights file.
        """

        self._classifier = self.load_classifier(classifier_weights)

    def _load_image(self, image):
        """Loads an image from array or string."""
        if type(image) is str:
            image = cv2.imread(image)

        if image is None:
            raise ImageCheck.NotAnImageError

        return image

    def load_classifier(self, mlp_weights_path):
        """ Loads classifier via saved weights.

        Args:
            weights_path (str): Path to the saved CNN classifier weights.
        Returns:
            The loaded fully connected and trained ResNet18 model.
        """
        return BC.make_classifier(mlp_weights_path)

    def process_image(self, image, dims=False):
        """Processes an image using imagecheck.check_and_crop function.

        Args:
            image (str or numpy.ndarray): Path to the image to be process.
        Returns:
            A cropped image containing the worm.
        Raises:
            ImageCheckError: Raised from `imagecheck/ImageCheck.py`
            if the image does not meet the required standards or no
            worm is found.
        """
        image = self._load_image(image)
        return ImageCheck.check_and_crop(image, dims)

    def predict(self, image, preprocessed=False):
        """Predict whether an image contains a Fall Armyworm.

        Args:
            image (str or numpy.ndarray): Path to the image containing object
                to be classified.
            preprocessed (bool): If True, no processing will be applied to the
                image as supplied image is expected to have already been
                processed to meet the format requirements..

        Returns:
            True if Fall Armyworm detected, else False.
        """
        image = self._load_image(image)
        if not preprocessed:
            image = self.process_image(image, IMG_DIMS)
        # reshape image to expected TF format (None, channels, height, width)
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)

        return self._classifier.predict(image)


def detect_fallarmyworm(image, threshold=0.5, preprocessed=False):
    """Predicts whether an iamge contains a Fall Armyworm or not.

    Args:
        image_path (str): path to the image
        threshold (double): Threhold above which the ResNet positive
        probability is classed as a positive classifiation. Default = 0.5
    Returns:
        bool : True is contains Fall Armyworm, False otherwise.
    """
    classifier = FAW_classifier()

    if classifier.predict(image, preprocessed) > threshold:
        return True

    return False


# Argparse options for running from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_img", type=str,
                        help=("path to the image to be analysed by the "
                              "Fall Armyworm detector"))
    args = parser.parse_args()
    classifier = FAW_classifier()
    if pathlib.Path(args.path_to_img).is_file() and args.path_to_img != '':
        prediction = classifier.predict(args.path_to_img)
        print("\nFall Armyworm detected probability: " + str(prediction))
    else:
        print("\nFAW_Detector ERROR: {} is not a valid file.".format(
            args.path_to_img))
