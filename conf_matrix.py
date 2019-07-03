from sklearn.metrics import classification_report, confusion_matrix
from FAW import FAW_classifier
from pathlib import Path


classifier = FAW_classifier.FAW_classifier()

path_to_valid = Path('data/validation')

faw_files = path_to_valid.glob('faw/*.jpg')
notfaw_files = path_to_valid.glob('notfaw/*.jpg')

total_faw = 0
total_notfaw = 0
rejected_faw = 0
rejected_notfaw = 0
correct_faw = 0
false_faw = 0
correct_notfaw = 0
false_notfaw = 0


for faw in faw_files:
    try:
        result = classifier.predict(faw)
    except Exception as e:
        print(e)
        rejected_faw += 1

    if result >= 0.5:
        correct_faw += 1
    else:
        false_faw += 1

for faw in faw_files:
    try:
        result = classifier.predict(faw)
    except Exception as e:
        print(e)
        rejected_notfaw += 1

    if result >= 0.5:
        correct_notfaw += 1
    else:
        false_notfaw += 1

print('total_faw = {total_faw}')
print('total_notfaw = {total_notfaw}')
print('rejected_faw = {rejected_faw}')
print('rejected_notfaw = {rejected_notfaw}')
print('correct_faw = {correct_faw}')
print('false_faw = {false_faw}')
print('correct_notfaw = {correct_notfaw}')
print('false_notfaw = {false_notfaw}')

