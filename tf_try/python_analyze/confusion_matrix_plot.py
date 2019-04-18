# coding: utf-8

import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None, save=None):
    figsize = (25, 20)
    if isinstance(cm, tuple):
        true_label, pred_label = cm
        cm = confusion_matrix(true_label, pred_label)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.size'] = 50
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=50, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=50)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=50)
    plt.xlabel('Predicted label', fontsize=50)
    if save:
        plt.savefig(save)
    plt.show()

class_names = [ '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11', '12', '13']
# class_names = ['Scrub','Willow swamp','CP hammock','CP/Oak','Slash pine','Oak/Broadleaf','Hardwood swamp','Graminoid marsh','Spartna marsh','Cattail marsh','Salt marsh','Mud flats','Water']

cm = np.array([[149,0,0,0,0,4,0,0,0,0,0,0,0],
[1,44,0,0,0,0,4,0,0,0,0,0,0],
[0,0,51,0,0,0,0,0,1,0,0,0,0],
[2,0,3,44,1,0,0,1,0,0,0,0,0],
[0,0,0,6,27,0,0,0,0,0,0,0,0],
[4,1,1,0,0,39,0,0,1,0,0,0,0],
[0,1,0,0,0,0,20,0,0,0,0,0,0],
[0,0,1,1,0,0,0,85,0,0,0,0,0],
[0,0,0,0,0,0,0,0,104,0,0,0,0],
[0,0,0,0,0,0,0,0,0,81,0,0,0],
[0,0,0,0,0,0,0,0,0,0,84,0,0],
[0,0,0,0,0,0,0,0,0,1,0,100,0],
[0,0,0,0,0,0,0,0,0,0,0,0,186],
])
plot_confusion_matrix(cm,class_names,title='Confusion matrix',save='test.png')
