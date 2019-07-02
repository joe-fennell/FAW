import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
from ast import literal_eval


def smooth_data(y):
    x = np.linspace(0, len(y), len(y))
    xx = np.linspace(0, len(y), len(y))

    # interpolate + smooth
    itp = interp1d(x, y, kind='linear')
    window_size, poly_order = 11, 2
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    return yy_sg


def plot_metric(train, val, title):
    plt.clf()
    plt.plot(train, label='train', color='blue', linestyle='--', alpha=0.5)
    plt.plot(val, label='val', color='red', linestyle='--', alpha=0.5)
    plt.plot(smooth_data(train), label='smooth train', color='blue', linestyle='-')
    plt.plot(smooth_data(val), label='smooth val', color='red', linestyle='-')
    if title == 'Accuracy':
        plt.ylim(0.75, 1.02)
        axes = plt.axes()
        axes.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    plt.title(title)
    plt.ylabel(title)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(title+'.png')

    return None


def load_data(filename):
    with open(filename) as f:
        history = json.load(f)
        #history = literal_eval(history)
    return history


def compare(data):
    train = np.average(data['acc'][-10:])
    val = np.average(data['val_acc'][-10:])
    diff = train - val
    print('Train     : ' + str(train))
    print('Validation: ' + str(val))
    print('Difference: ' + str(diff))
    return None


history = load_data('../0002_finetuning_history_amsgrad_lr00001.json') #change this to your file
compare(history)
plot_metric(history['loss'], history['val_loss'], 'Loss')
plot_metric(history['acc'], history['val_acc'], 'Accuracy')

