""" Cleaning script template."""

import json
import numpy as np

def load_data(path):
    # load in the data
    with open(path, 'r') as fp:
        x = json.load(fp)
    return x

def save_data(path, data):
    # load in the data
    with open(path, 'w') as fp:
        json.dump(data, fp)

worms = load_data('worm_shapefactors.json')
not_worms = load_data('other_shapefactors.json')

print(len(not_worms))

save_data('other_shapefactors.json', not_worms)
save_data('worm_shapefactors.json', worms)




