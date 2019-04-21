""" Test script for identifying worm contours from shape factors."""

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# do mean normalisation
x_norm = np.mean(x_train, axis=0)
x_train = x_train - x_norm
x_test = x_test - x_norm

best_score = 0
best_average = 0
best_j = 0
best_j_k = ''

for j in range(1,7):

    pca = PCA(n_components=j)
    x_fit = pca.fit_transform(x_train)
    x_fit_test = pca.transform(x_test)

    scores_total = 0

    for k in range(1,26):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(x_fit, y_train)

        score = knn.score(x_fit_test, y_test)
        
        if score > best_score:
            best_score = score
            best_j_k = str(j) + ' ' + str(k)

        scores_total = scores_total + score

    scores_average = scores_total / 25

    if scores_average > best_average:
        best_average = scores_average
        best_j = j

print('Best PCA: {}, {}'.format(best_j, best_average))
print('Best overall: {}, {}'.format(best_j_k, best_score))




