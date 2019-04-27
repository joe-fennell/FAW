"""
Fall Armyworm Project - University of Manchester.

clean_and_crop.py
Contributor: George Worrall

Script to run existing training and validation data through
imagecheck/imagecheck.py to filter and crop data.
"""

import pathlib 
import imagecheck

data_dirs = ['train/', 'validation/']
categories = ['faw/', 'notfaw/']
new_dirs = ['train_cropped/', 'validation_cropped']

for data_dir in data_dirs:
    for category in categories:
        loc_str = '../FAW/data/' + data_dir + category
        path = Path(loc_str)
        imgs = list(path.glob('*.jpg'))
        for img in imgs:
            print(img)



