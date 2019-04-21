""" Test script for identifying worm contours from shape factors."""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def load_data(path):
    # load in the data
    with open(path, 'r') as fp:
        x = json.load(fp)
    return x

not_worms = load_data('../sf_data/other_shapefactors.json')
worms = load_data('../sf_data/worm_shapefactors.json')

# generate labels
worm_labels = [1] * len(worms)
not_worm_labels = [0] * len(not_worms)

# combine the data
x = worms + not_worms
y = worm_labels + not_worm_labels
    
# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

# train SCV classifier
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)

train_score = svc.score(x_train, y_train)

test_score = svc.score(x_test, y_test)

print(train_score)
print(test_score)

