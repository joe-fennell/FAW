"""
Fall Armyworm Project - University of Manchester.

clean_and_crop.py
Contributor: George Worrall

Script to run existing training and validation data through
imagecheck/imagecheck.py to filter and crop data.
"""

import pathlib 
import imagecheck
import cv2 as cv

data_dirs = ['train/', 'validation/']
categories = ['faw/', 'notfaw/']
new_dirs = ['train_cropped/', 'validation_cropped']

for data_dir in data_dirs:
    for category in categories:
        loc_str = '/mnt/data/' + data_dir + category
        path = pathlib.Path(loc_str)
        imgs = list(path.glob('*.jpg'))
        for img in imgs:
            print(img)
            try:
                cropped_img = imagecheck.crop(str(img))
            except imagecheck.ImageCheckError as e:
                continue  # skip ahead for images that don't pass
            if data_dir is 'train':
                new_dir = new_dirs[0]
            else:
                new_dir = new_dirs[1]
            new_loc = '/mnt/data/' + new_dir + category + img.name
            cv.imwrite(new_loc, cropped_img)




