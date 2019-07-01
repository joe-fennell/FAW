"""
Fall Armyworm Project - University of Manchester
Author: George Worrall

imagecheck.py

Script to segment worm foreground objects from (mostly) uniform background.

Outputs cropped image from original containing only the detected worm.

For images with multiple foreground objects, script uses trained MLP to guess
which is the worm by computing size-independent shape factors.
Also performs blur check on images to ensure they are above a predefined blur
threshold.
"""

# TODO: consider drawing box on live feed on phone and asking users to take a
# picture with the caterpillar inside the specific box. Reject image if any
# contours outside the box (would require uniform background).

# TODO: look at report structure etc for something to give out on the project
# on method, differenty design options considered etc.

import pickle
import math
import pathlib
import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

# Globals
SCALED_IMAGE_PIXELS = 50176  # 224 x 224 image pixels
MORPH_KERNAL_RATIO = 0.01
BASE_PATH = str(pathlib.Path(__file__).parent)


def _get_colour_codes(arr):

    # returns the foreground and background colour codes
    # NOTE: this is based on the assumption that foreground will take up less
    # than 50% of the pixels in the supplied picture

    codes, counts = np.unique(arr, return_counts=True)
    # for some uniform pictures with only background, the count may be 1
    # if so, reject
    if len(counts) != 2:
        return False, False

    if counts[0] > counts[1]:
        return codes[1], codes[0]

    return codes[0], codes[1]


def _downscale_image(img, scale=1, dims=False):

    # downscales the image to make k means less resource intensive

    if dims:
        return cv2.resize(img, dsize=dims, interpolation=cv2.INTER_AREA)

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized, height, width


def _blur_check(img):

    # Use laplacian blur detection to reject blurry images

    blur_var = cv2.Laplacian(img, cv2.CV_64F).var()

    if blur_var < 1.5:
        print(blur_var)
        return False

    return True


def _get_shape_factors(cnt):

    # Calculate shape factor properties of a contour
    # https://docs.opencv.org/4.0.1/d1/d32/tutorial_py_contour_properties.html
    # https://en.wikipedia.org/wiki/Shape_factor_(image_analysis_and_microscopy)
    # https://www.researchgate.net/publication/
    # 225405567_Measuring_Elongation_from_Shape_Boundary

    # aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w)/h

    # extent
    area = cv2.contourArea(cnt)
    rect_area = w * h
    extent = float(area)/rect_area

    # solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area

    # circularity
    perimiter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimiter ** 2)

    # elongation
    m = cv2.moments(cnt)
    j = m['mu20'] + m['mu02']
    k = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    elongation = (j + k**0.5) / (j - k**0.5)

    # compactness
    compactness = area**2 / (2 * np.pi * np.sqrt(j**2 + k**2))

    return [aspect_ratio, extent, solidity, circularity,
            elongation, compactness]


def _plot_contours(img, contours, h, w, scale_ratio):

    # plots bounding boxes of contours on an image for debugging

    print("Contour count: {}".format(len(contours)))

    for contour in contours:
        scld = _scale_dims(contour, img, scale_ratio)
        cv2.rectangle(img, (scld[0], scld[1]),
                      (scld[0]+scld[2], scld[1]+scld[3]),
                      (255, 0, 0), 4)

    cv2.namedWindow('check', cv2.WINDOW_NORMAL)
    cv2.imshow('check', img)
    cv2.resizeWindow('check', 800, 800)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _scale_dims(contour, img, scale_ratio):

    # scale up dimensions according to scale ratio

    br_dims = cv2.boundingRect(contour)
    scld = [int(i / scale_ratio) for i in br_dims]
    # add 2% buffer to prevent minor clipping of AOI
    (h, w) = img.shape[:2]
    scld[0] = scld[0] - int(2 * w / 100)  # x value base
    scld[1] = scld[1] - int(2 * h / 100)  # y value base
    scld[2] = scld[2] + int(4 * w / 100)  # x width
    scld[3] = scld[3] + int(4 * h / 100)  # y width
    scld = [0 if x < 0 else x for x in scld]  # no neg buffer pixel values

    return scld


def _contour_sorting(contours, hierarchy, pixels, h, w):
    """Finds the worm contour from a list of contours and a hierarchy using MLP
    classifier..
        - Rejects contours of less than 1% total pixel area.
        - Rejects any contours which are touching the image border."""

    # get all parents contours in heirarchy
    # https://opencv-python-tutroals.readthedocs.io/en/latest
    # /py_tutorials/py_imgproc/py_contours/py_contours_hierarchy
    # /py_contours_hierarchy.html
    hierarchy = hierarchy.tolist()
    parent_contours = [contours[hierarchy[0].index(x)]
                       for x in hierarchy[0] if x[3] == -1]

    # ignore any contours that touch the image border or are too small
    contours_accepted = []
    for contour in parent_contours:
        if cv2.contourArea(contour) < (pixels/100):
            # ignore contours of > 1% pixel area
            continue
        x_coords = contour[:, :, 0].flatten()
        y_coords = contour[:, :, 1].flatten()
        if (0 in x_coords) or (0 in y_coords):  # reject border pixel values
            continue
        if (w - 1) in x_coords:
            continue
        if (h - 1) in y_coords:
            continue
        contours_accepted.append(contour)

    if len(contours_accepted) == 0:
        raise ObjectMissingError("No suitable foreground objects found.")

    # NOTE: if only one parent contour, it is assume to be the worm
    if len(contours_accepted) == 1:
        return contours_accepted[0]

    raise MultipleWormsError("More than one potential worm contour found.")


def check_and_crop(img_arg, dims=False):
    """Finds parent contours in an image, gets shape factors for those contours
    then crops the image to the area of interest containing the worm.

    If image given is larger than 224 x 224 or equivalent, scales the image
    down for preprocesing and then crops the original.

    Contour detection notes:
        - If a contour touches the image border, it is rejected.
        - If a contour is less than 1% of total image area, it is rejected.

    Checks the image to see if it conforms to:
        - A preset blur threshold

    Args:
        img_arg (str or numpy.ndarray): String containing the location of the
            image file.
        dims (tuple): Required height and width for the image in format
        (height, width)

    Returns:
        numpy.ndarray: Contains the cropped image as a an array of shape
        (h, w, 3).

    Raises:
        ObjectMissingError: No foreground objects found in image.
        WormMissingError: No worm found in image.
        MultipleWormsError: Two or more worms found in image.
        TooBlurryError: Image too blurry.
    """

    # load from string if file path passed rather than ndarray
    img = img_arg
    if type(img_arg) is str:
        img = cv2.imread(img_arg)

    # initial blur check
    if not _blur_check(img):
        raise TooBlurryError("Image too blurry.")

    # Save original image copy and downscale for k means
    img_copy = img.copy()
    (h, w) = img.shape[:2]
    pixels = h * w
    scale_ratio = 1
    if pixels > SCALED_IMAGE_PIXELS:  # 224 x 224 or similar dims size
        # sqrt for dims scale ratio
        scale_ratio = math.sqrt(SCALED_IMAGE_PIXELS / pixels)
        img, h, w = _downscale_image(img, scale_ratio)
        pixels = h * w

    # go to LAB place for k means and reshape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # k means cluster to find two dominant colour groups
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape back to image dimensions
    quant = quant.reshape((h, w, 3))

    # come out of LAB space
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    quant = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)

    plt.imshow(quant)
    plt.show()

    # get foreground objects colour code
    fg_code, bg_code = _get_colour_codes(quant)

    # reject if codes could not be found, ie. picture is uniform colour
    if not fg_code:
        raise ObjectMissingError("Could not locate any foreground objects.")

    if fg_code > bg_code:
        ret, thresh = cv2.threshold(quant, fg_code - 1, 1,
                                    cv2.THRESH_BINARY)
    else:
        ret, thresh = cv2.threshold(quant, fg_code + 1, 1,
                                    cv2.THRESH_BINARY_INV)

    # perform a small kernel closing operation to smooth noise around contour
    # edges - kernal size based on downscaled image dimensions
    kernel = np.ones((int(h * MORPH_KERNAL_RATIO),
                      int(h * MORPH_KERNAL_RATIO)), np.uint8)
    img_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get contours
    contours, hierarchy = cv2.findContours(img_closed.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    worm_contour = _contour_sorting(contours, hierarchy, pixels, h, w)

    # crop image to selection - need to upscale values to original dims

    scld = _scale_dims(worm_contour, img_copy, scale_ratio)

    crop_img = img_copy[scld[1]:scld[1]+scld[3], scld[0]:scld[0]+scld[2]]

    if dims:
        crop_img = _downscale_image(crop_img, dims=dims)

    return crop_img


class ObjectMissingError(Exception):
    """Error raised when no foreground objects are detected in an image."""
    pass


class WormMissingError(Exception):
    """Error raised when no worm/caterpillar like object found in an image."""
    pass


class MultipleWormsError(Exception):
    """Error raised when more than one potential worm contour found in an
    image."""
    pass


class TooBlurryError(Exception):
    """Error raised when an image is too blurry."""
    pass
