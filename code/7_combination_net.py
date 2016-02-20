from __future__ import division
import h5py
import time as tm
import numpy as np
import pandas as pd
from copy import copy

from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.regularizers import l2
from keras.callbacks import Callback

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

weights_path = '/data/vgg16_weights.h5'
training_input = '/data/X_artists.npy'   # X_genres.npy    X_locations.npy
training_output = '/data/Y_artists.npy'  # Y_genres.npy    Y_locations.npy
img_width, img_height = 128, 128
epoch_count = 14
learning_rates = [7.4e-5, 4.2e-5, 1.2e-5]
regularization_strengths = 10**np.random.uniform(-7,-3,1)


X = np.load(training_input).astype('float32')
y = np.load(training_output).astype('float32')
X -= np.mean(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
