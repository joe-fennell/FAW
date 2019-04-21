"""
segmentation.py

Script to segment foreground objects from (mostly) uniform background with
variable lighting. Also detects and rejects images with multiple foreground
objects.
"""

# TODO: consider drawing box on live feed on phone and asking users to take a
# picture with the caterpillar inside the specific box. Reject image if any
# contours outside the box (would require uniform background).

# TODO: look at real time identification in phone before picture is taken

# TODO: CONSIDER recalucating shape factors with very minimal erosion and
# dilation to retain more of the orignal shape. May help to differentiate
# between worms and non worms

# TODO: look at low cost CNNs and other options for classifying if contour is a
# caterpillar. Possibly faster R CNN


import glob
import json
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans


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


def _downscale_image(img, scale):

    # downscales the image to make k means less resource intensive

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    return resized, height, width


def _blur_check(img):

    # Use laplacian blur detection to reject blurry images

    blur_var = cv.Laplacian(img, cv.CV_64F).var()

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
    x, y, w, h = cv.boundingRect(cnt)
    aspect_ratio = float(w)/h

    # extent
    area = cv.contourArea(cnt)
    rect_area = w * h
    extent = float(area)/rect_area

    # solidity
    hull = cv.convexHull(cnt)
    hull_area = cv.contourArea(hull)
    solidity = float(area)/hull_area

    # circularity
    perimiter = cv.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimiter ** 2)

    # elongation
    m = cv.moments(cnt)
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
        br_dims = cv.boundingRect(contour)
        scld = [int(i * scale_ratio) for i in br_dims]
        # add 2% buffer to prevent minor clipping of AOI
        (h, w) = img.shape[:2]
        scld[0] = scld[0] - int(2 * w / 100)  # x value base
        scld[1] = scld[1] - int(2 * h / 100)  # y value base
        scld[2] = scld[2] + int(4 * w / 100)  # x width
        scld[3] = scld[3] + int(4 * h / 100)  # y width
        scld = [1 if x < 0 else x for x in scld]  # no neg buffer pixel values
        cv.rectangle(img, (scld[0], scld[1]),
                     (scld[0]+scld[2], scld[1]+scld[3]), (255, 0, 0), 4)

    cv.namedWindow('check', cv.WINDOW_NORMAL)
    cv.imshow('check', img)
    cv.resizeWindow('check', 800, 800)
    cv.waitKey(0)
    cv.destroyAllWindows()


def check_image(img_location):

    # TODO: edit this description when finished
    """Finds parent contours in an image, gets shape factors for those contours
    then crops the image to the area of interest containing the worm.

    If image given is larger than 800 x 800 or equivalent, scales the image
    down for preprocesing and then crops the original.

    Contour detection notes:
        - If a contour touches the image border, it is rejected.
        - If a contour is less than 1% of total image area, it is rejected.

    Checks the image to see if it conforms to:
        - A preset blur threshold
    """

    img = cv.imread(img_location)

    # Save original image copy and downscale for k means
    img_copy = img.copy()
    (h, w) = img.shape[:2]
    pixels = h * w
    scale_ratio = 1
    if pixels > 640000:  # 800 x 800 or similar dims size
        scale_ratio = 640000 / pixels
        img, h, w = _downscale_image(img, scale_ratio)
        pixels = h * w

    # go to LAB place for k means and reshape
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # k means cluster to find two dominant colour groups
    clt = MiniBatchKMeans(n_clusters=2)
    labels = clt.fit_predict(img)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape back to image dimensions
    quant = quant.reshape((h, w, 3))

    # come out of LAB space
    quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
    quant = cv.cvtColor(quant, cv.COLOR_BGR2GRAY)

    # get foreground objects colour code
    fg_code, bg_code = _get_colour_codes(quant)

    # reject if codes could not be found, ie. picture is uniform colour
    if not fg_code:
        return False

    if fg_code > bg_code:
        ret, thresh = cv.threshold(quant, fg_code - 1, 1, cv.THRESH_BINARY)
    else:
        ret, thresh = cv.threshold(quant, fg_code + 1, 1, cv.THRESH_BINARY_INV)

    # perform a small kernel closing operation to smooth noise around contour
    # edges - kernal size based on downscaled image dimensions
    kernel = np.ones((int(h/100), int(h/100)), np.uint8)
    img_closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # get contours
    contours, hierarchy = cv.findContours(img_closed.copy(), cv.RETR_TREE,
                                          cv.CHAIN_APPROX_SIMPLE)

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
        if cv.contourArea(contour) < (pixels/100):
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

    # reject if more than one foreground object or 0 found
    if len(contours_accepted) > 1:
        print("IMAGE REJECTED")
        # _plot_contours(img_copy, parent_contours, h, w, scale_ratio)
        # TODO: remove plot contours once finished
        return False
    if len(contours_accepted) == 0:
        print("IMAGE border REJECTED")
        # _plot_contours(img_copy, parent_contours, h, w, scale_ratio)
        # TODO: remove plot contours once finished
        return False

    # TODO: set up combined factor extraction and classification
    # factors = _get_shape_factors(contours_accepted[0])
    # global factors_list
    # factors_list.append(factors)

    # crop image to selection - need to upscale values to original dims
    br_dims = cv.boundingRect(contours_accepted[0])
    scld = [int(i / scale_ratio) for i in br_dims]
    # add 2% buffer to prevent minor clipping of AOI
    (h, w) = img_copy.shape[:2]
    scld[0] = scld[0] - int(2 * w / 100)  # x value base
    scld[1] = scld[1] - int(2 * h / 100)  # y value base
    scld[2] = scld[2] + int(4 * w / 100)  # x width
    scld[3] = scld[3] + int(4 * h / 100)  # y width

    scld = [0 if x < 0 else x for x in scld]  # no negative buffer pixel values

    crop_img = img_copy[scld[1]:scld[1]+scld[3], scld[0]:scld[0]+scld[2]]

    # blur check
    if not _blur_check(crop_img):
        print("Image blur rejected.")
        return False

    return crop_img

#  Get all test files from worms dir.
#
# imgs = list(glob.glob('worms/**/*.jpg'))
#
# factors_list = []
# i = 1
# for img in imgs:
#     print(img)
#     print('{} / {}'.format(i, len(imgs)))
#     check_image(img)
#     print("Factor collections: {}".format(len(factors_list)))
#     i += 1
#
# with open("worm_shapefactors.json", 'w') as fp:
#     json.dump(factors_list, fp)
