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
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.callbacks import Callback

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

env = 'local'
if env == 'local':
  path = '/Users/derekchen/Documents/conv_nets/cs231n'
  weights_path = path+'/data/vgg16_weights.h5'
  training_input = path+'/data/X.npy'
  training_output = path+'/data/Y.npy'
elif env == 'remote':
  weights_path = '/data/vgg16_weights.h5'
  training_input = '/data/X_artists.npy'
  training_output = '/data/Y_artists.npy'

X = np.load(training_input).astype('float32')
y = np.load(training_output).astype('float32')
X -= np.mean(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

if env == 'local':
  classes = 3
  rates = [7.4e-5, 1.2e-5, 4.4e-6]
  n_trials = 3
  epoch_count = 3
  img_width, img_height = 128, 128
  X_train, X_test = X_train[:50,:,:,:], X_test[:50,:,:,:]
  y_train, y_test = y_train[:50,:], y_test[:50,:]
  train_level = 0
elif env == 'remote':
  classes = 20
  rates = [7.4e-5, 4.2e-5, 1.2e-5, 7.4e-6, 4.4e-6]
  n_trials = 25
  epoch_count = 14
  img_width, img_height = 224, 224
  train_level = 0

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class ModelMaker(object):

  def __init__(self,hyperparams,train_level):
    self.learning_rate = hyperparams[0]
    self.reg_strength = hyperparams[1]
    self.dropout_prob = round(hyperparams[2], 2)
    self.batch_history = None
    self.epoch_history = None
    self.model = None
    self.train_level = train_level
    print 'Learning rate is', self.learning_rate
    print 'Reg_strength is', self.reg_strength
    print 'Dropout_prob is', self.dropout_prob

  def createModel(self):
    # train_level = 0: only train FC layers
    #         1: train FC layers and last 3 conv layers
    #         2: train FC layers and last 6 conv layers

    # build the VGG16 network with our input_img as input

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if env == 'remote':
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

      for layer in model.layers:
        layer.trainable = False

      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv4_1'))
      if train_level == 2: model.add(BatchNormalization())
      model.add(Activation('relu',name='relu_1'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv4_2'))
      if train_level == 2: model.add(BatchNormalization())
      model.add(Activation('relu',name='relu_2'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv4_3'))
      if train_level == 2: model.add(BatchNormalization())
      model.add(Activation('relu',name='relu_3'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))

      if train_level < 2:
        for layer in model.layers:
          layer.trainable = False

      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_1'))
      if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_2'))
      if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_3'))
      if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))

      if train_level < 1:
        for layer in model.layers:
          layer.trainable = False

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # Load the weights from our dropbox folder (about 0.5 GB worth) --------------------------
    f = h5py.File(weights_path)
    skipped = 0
    for k in range(f.attrs['nb_layers']):
      if k >= len(model.layers):
        break         # we don't look at the last (fully-connected) layers in the savefile
      isActivation = (type(model.layers[k]) == type(Activation('relu')))
      isBatchNorm = (type(model.layers[k]) == type(BatchNormalization()))
      if isActivation | isBatchNorm:
        skipped += 1
        print 'skipping'
        continue #skip activation layers
      g = f['layer_{}'.format(k-skipped)]
      print g.keys()
      weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
      print type(model.layers[k])
      model.layers[k].set_weights(weights)
    f.close()
    print('Model weights loaded.')

    model.add(Flatten())
    # Note: Keras does automatic shape inference.

    if env == 'local':
      model.add(Dense(256,name='dense_1',init='he_normal',W_regularizer=l2(self.reg_strength)))
      model.add(Activation('relu'))
      model.add(Dropout(self.dropout_prob,name='dropout_1'))

    if env == 'remote':
      model.add(Dense(4096,name='dense_1',init='he_normal',W_regularizer=l2(self.reg_strength)))
      model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(Dropout(self.dropout_prob,name='dropout_1'))

      model.add(Dense(4096,name='dense_2',init='he_normal',W_regularizer=l2(self.reg_strength)))
      model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(Dropout(self.dropout_prob,name='dropout_2'))

      model.add(Dense(1000,name='dense_3',init='he_normal',W_regularizer=l2(self.reg_strength)))
      model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(Dropout(self.dropout_prob,name='dropout_3'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    self.model = model

  def run(self):
    adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    initial_time = tm.time()
    self.model.compile(loss='categorical_crossentropy', optimizer=adam)
    checkpoint = tm.time() - initial_time
    print 'Compiled in %s seconds' % round(checkpoint, 3)

  def fit_data(self):
    batch_history = LossHistory()
    epoch_history = self.model.fit(X_train, y_train, batch_size=32, nb_epoch=epoch_count,
      verbose=1, show_accuracy=True, callbacks=[batch_history], validation_split=0.2)

    last_loss = epoch_history.history['val_loss'][-1]
    last_acc = epoch_history.history['val_acc'][-1]
    print 'Last validation loss for this iteration is', round(last_loss,4)
    print 'Last validation accuracy is', round(last_acc,4)

    self.batch_history = batch_history
    self.epoch_history = epoch_history


class CrossValidator(object):
  def __init__(self):
    self.batch_histories = []
    self.epoch_histories = []
    self.epoch_acc_histories = []
    self.best_model = None
    self.best_val_loss = 1e9
    self.best_model_params = {}

  def plot(self):
      plt.figure()
      plt.xlabel('Batch Number')
      plt.ylabel('Training Loss')
      for history in self.batch_histories:
        plt.plot(history)
      plt.savefig('Feb_train_loss.png')

      plt.figure()
      plt.xlabel('Epoch Number')
      plt.ylabel('Validation Loss')
      for history in self.epoch_histories:
        plt.plot(history)
      plt.savefig('Feb_val_loss.png')

      plt.figure()
      plt.xlabel('Epoch Number')
      plt.ylabel('Validation Accuracy')
      for history in self.epoch_acc_histories:
        plt.plot(history)
      plt.savefig('Feb_val_accuracy.png')
      history_dict = {'batch_histories': self.batch_histories, 'epoch_histories': self.epoch_histories, 'epoch_acc_histories': self.epoch_acc_histories}
      pickle.dump(history_dict,open('history_dict.out','wb'))

  def update(self,src,iteration):
    if iteration == 0:
      print 'first iteration; saving model'
      self.best_model = copy(model)
      self.best_val_loss = model.epoch_history.history['val_loss'][-1]
      self.best_model_params = {'learning_rate': model.learning_rate,
       'reg_strength': model.reg_strength, 'dropout_prob': model.dropout_prob}

    elif epoch_history.history['val_loss'][-1] < self.best_val_loss:
      print 'I think this current model is better: Im saving it.'
      self.best_model = copy(model)
      self.best_val_loss = model.epoch_history.history['val_loss'][-1]
      self.best_model_params = {'learning_rate': model.learning_rate,
        'reg_strength': model.reg_strength, 'dropout_prob': model.dropout_prob}

      self.batch_histories.append(model.batch_history.losses)
      self.epoch_histories.append(model.epoch_history.history['val_loss'])
      self.epoch_acc_histories.append(model.epoch_history.history['val_acc'])

def build_ensembles(hyperparams_list):
  ensemble_results = []
  solver = CrossValidator()

  for trial in range(n_trials):

    maker = ModelMaker(hyperparams[trial], train_level)
    print 'Running crossval trial', trial+1
    maker.createModel()
    maker.run()

    solver.update(maker,trial)
    test_predictions = model.predict(X_test,batch_size=32,verbose=1)
    print "Test Accuracy:"
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

  for i in range(X_train.shape[0]):
    predictions[i, answers[i]] = 1.
  return predictions


learning_rates = 10**np.random.uniform(-6,-3,n_trials).astype('float32')
reg_strengths = 10**np.random.uniform(-7,0,n_trials).astype('float32')
dropout_probs = np.random.uniform(0.1,0.9,n_trials).astype('float32')
hyperparams = zip(learning_rates,reg_strengths,dropout_probs)

ensemble_results = build_ensembles(hyperparams)
final_predictions = vote_for_best(ensemble_results)
print 'Final accuracy is', np.sum(np.argmax(final_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]
print "Sixth net is done."