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
training_input = '/data/X_artists.npy'
training_output = '/data/Y_artists.npy'
img_width, img_height = 128, 128
epoch_count = 14
learning_rates = [7.4e-5, 4.2e-5, 1.2e-5]
num_networks_in_ensemble = 5

X = np.load(training_input).astype('float32')
y = np.load(training_output).astype('float32')
X -= np.mean(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

def createModel():
  # build the VGG16 network with our input_img as input
  first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))

  model = Sequential()
  model.add(first_layer)

  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
  model.add(ZeroPadding2D((1, 1)))
  model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))

  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])

  # Load the weights from our dropbox folder (about 0.5 GB worth) --------------------------
  f = h5py.File(weights_path)

  for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
      break         # we don't look at the last (fully-connected) layers in the savefile
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
  f.close()
  print('Model loaded.')

  # Leave the pretrained layers untouched -----------------
  for layer in model.layers:
    layer.trainable = False
  return model

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class CrossValidator(object):

  def __init__(self):
    self.batch_histories = []
    self.epoch_histories = []
    self.epoch_acc_histories = []
    self.best_model = None
    self.best_val_loss = 1e9

  def run(self,rate):
    model = createModel()

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(4096,name='dense_1',init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,name='dropout_1'))

    model.add(Dense(4096,name='dense_2',init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,name='dense_2',init='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5,name='dropout_1'))

    model.add(Dense(20))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return self.solve_model(optimizer)

  def solve_model(self, optimizer):
    initial_time = tm.time()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    checkpoint = tm.time() - initial_time
    print 'Compiled in %s seconds' % round(checkpoint, 3)
    return self.fit_data(model)

  def fit_data(self,model):
    batch_history = LossHistory()
    epoch_history = model.fit(X_train, y_train, batch_size=32, nb_epoch=epoch_count, verbose=1,
      show_accuracy=True, callbacks=[batch_history], validation_split=0.2)

    last_loss = epoch_history.history['val_loss'][-1]
    last_acc = epoch_history.history['val_acc'][-1]
    print 'Last validation loss for this iteration is', round(last_loss,4) , 'current best is', self.best_val_loss
    print 'Last validation accuracy is', round(last_acc,4)

    if iteration == 0:
      print 'first iteration; saving model'
      self.best_model = copy(model)
      self.best_val_loss = epoch_history.history['val_loss'][-1]
    elif epoch_history.history['val_loss'][-1] < self.best_val_loss:
      print 'I think this current model is better: Im saving it.'
      self.best_model = copy(model)
      self.best_val_loss = epoch_history.history['val_loss'][-1]

    self.batch_histories.append(batch_history.losses)
    self.epoch_histories.append(epoch_history.history['val_loss'])
    self.epoch_acc_histories.append(epoch_history.history['val_acc'])
    return model

def build_ensembles(learning_rates):
  ensemble_results = []

  for rate in learning_rates:
    solver = CrossValidator()  # should really be renamed as ModelMaker
    model = solver.run(rate)
    print "Test Accuracy for Learning Rate", rate
    test_predictions = model.predict(X_test,batch_size=32,verbose=1)
    print np.sum(np.argmax(test_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]
    ensemble_results.append(test_predictions)

  return ensemble_results

def vote_for_best(results):
  answers = np.zeros(results[0].shape)
  predictions = np.zeros(results[0].shape)

  for idx, result in enumerate(results):
    weighting = 1+(idx/100.0)
    weighted_results = result * weighting
    answers += weighted_results
  answers = np.argmax(answers, axis=1)

  for i in range(len(X_train[0])):
    predictions[i, answers[i]] = 1.
  return predictions


ensemble_results = build_ensembles(num_networks_in_ensemble)
final_predictions = vote_for_best(ensemble_results)
print 'Final accuracy is', np.sum(np.argmax(final_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]

np.save('epoch_losses.npy', np.array(solver.epoch_histories))
np.save('test_pred.npy',test_predictions)
np.save('train_pred.npy',train_predictions)
print "Sixth net is done."