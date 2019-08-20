"""
Testing script for model that has been training on the first data set.

Testing for performance on second data set.
"""

from FAW import FAW_classifier
from pathlib import Path

# Get all the images stored in the second data set
second_set_faw = list(Path('/home/gworrall/Documents/side_tasks/'
                         'faw_manc/FAW/data/sorted/second_set/faw'
                        ).glob('**/*.jpg'))
second_set_notfaw = list(Path('/home/gworrall/Documents/side_tasks/'
                         'faw_manc/FAW/data/sorted/second_set/notfaw'
                        ).glob('**/*.jpg'))

num_faw = len(second_set_faw)
num_notfaw = len(second_set_notfaw)

classifier = FAW_classifier.FAW_classifier()

true_count = 0
false_count = 0

i = 1
for worm in second_set_faw:
    print('{} / {}'.format(i, len(second_set_faw)))
    # skip images that fail the image check
    try:
        out = classifier.predict(str(worm), preprocessed=False)
    except Exception as e:
        print(e)
        i += 1
        continue
    if out > 0.5:
        true_count += 1
    else:
        false_count += 1
    i += 1
i = 1
for worm in second_set_notfaw:
    print('{} / {}'.format(i, len(second_set_notfaw)))
    # skip images that fail the image check
    try:
        out = classifier.predict(str(worm), preprocessed=False)
    except Exception as e:
        print(e)
        i += 1
        continue
    if out <= 0.5:
        true_count += 1
    else:
        false_count += 1
    i += 1

print(true_count / (true_count + false_count))

