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
from keras.preprocessing.image import ImageDataGenerator

from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils, generic_utils

env = 'full'
# path = '/Users/derekchen/Documents/conv_nets/cs231n'
path = ''
weights_path = path+'/data/vgg16_weights.h5'
classes = 20
n_trials, chunks = 2, 4
epoch_count = 4
img_width, img_height = 224, 224
batch_size = 10

def generate_hyperparams(n_trials):
  learning_rates = [2e-5, 7e-5]
  reg_strengths = [8e-4, 8e-4]
  dropout_probs = [0.4, 0.4]
  return zip(learning_rates,reg_strengths,dropout_probs)

def create_chunks(self):
  print "Loading data ..."
  initial_time = tm.time()
  X_path = '/Users/derekchen/Documents/conv_nets/cs231n/data/X_artists.npy'
  y_path = '/Users/derekchen/Documents/conv_nets/cs231n/data/Y_artists.npy'
  X = np.load(X_path).astype('float32')
  y = np.load(y_path).astype('float32')
  checkpoint = tm.time()
  print "Loaded in %0.2fs" % (checkpoint - initial_time)

  print X.shape
  print y.shape

  print "Shuffling data ..."
  X, y = shuffle(X, y, random_state=42)
  print "Shuffled in %0.2fs" % (tm.time() - checkpoint)

  print "Generating files ..."
  for j in xrange(X.shape[0]):
    jstr = str((j/500)+1)
    if j%500 == 0:
      if j < 5001:
        np.save('aX_chunk'+jstr+'of12.npy', X[j:j+500, :, :, :])
        np.save('ay_chunk'+jstr+'of12.npy', y[j:j+500])
      else:
        np.save('aX_chunk'+jstr+'of12.npy', X[j:, :, :, :])
        np.save('ay_chunk'+jstr+'of12.npy', y[j:])
      print "Finished Chunk "+jstr
  print "Done!"

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class ModelMaker(object):

  def __init__(self, hyperparams):
    self.learning_rate = hyperparams[0]
    self.reg_strength = hyperparams[1]
    self.dropout_prob = round(hyperparams[2], 2)
    self.batch_history = None
    self.epoch_history = None
    self.model = None
    self.trial = str(trial)
    print 'Learning rate is', self.learning_rate
    print 'Reg_strength is', self.reg_strength
    print 'Dropout_prob is', self.dropout_prob

  def create_model(self):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))
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
    model.add(Convolution2D(512, 3, 3, name='conv4_1'))
    model.add(Activation('relu',name='relu_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, name='conv4_2'))
    model.add(Activation('relu',name='relu_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, name='conv4_3'))
    model.add(Activation('relu',name='relu_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if env == 'full':
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_1'))
      # if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_2'))
      # if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(ZeroPadding2D((1, 1)))
      model.add(Convolution2D(512, 3, 3, name='conv5_3'))
      # if train_level == 1: model.add(BatchNormalization())
      model.add(Activation('relu'))
      model.add(MaxPooling2D((2, 2), strides=(2, 2)))

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
        # print 'skipping'
        continue #skip activation layers
      g = f['layer_{}'.format(k-skipped)]
      # print g.keys()
      weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
      # print type(model.layers[k])
      model.layers[k].set_weights(weights)
    f.close()
    print('Model weights loaded.')

    model.add(Flatten())

    if env == 'part':
      model.add(Dense(256,name='dense_1',init='he_normal',W_regularizer=l2(self.reg_strength)))
      model.add(Activation('relu'))
      model.add(Dropout(self.dropout_prob,name='dropout_1'))

    if env == 'full':
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

  def compile_model(self):
    adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    initial_time = tm.time()
    self.model.compile(loss='categorical_crossentropy', optimizer=adam)
    checkpoint = tm.time() - initial_time
    print 'Compiled in %s seconds' % round(checkpoint, 3)

  def fit_data(self):
    # batch_history = LossHistory()
    for e in range(epoch_count):
      print '>>> Epoch ', e+1
      # epoch_history = None
      for j in xrange(chunks):
        jstr = str(j+1)
        fp = path+'/data/chunks/'
        print 'Loading Chunk '+jstr

        X = np.load(fp+'aX_chunk'+jstr+'of12.npy')
        y = np.load(fp+'ay_chunk'+jstr+'of12.npy')
        print "finished loading chunks"


        print X.shape
        print y.shape
        X_train, y_train = self.preprocess_data(X, y)
        print X_train.shape
        print X[3, 2, :4,:4]
        print X_train[3, 2, :4,:4]
        print X_train[503, 2, :4,220:]
        print y_train.shape
        print "About to drop"

        tm.sleep(27)
        X = None
        y = None
        print "Finised dropping"
        tm.sleep(27)

        epoch_history = self.model.fit(X_train, y_train, nb_epoch=1,
          batch_size=batch_size, verbose=1, show_accuracy=True,
          validation_split=0.2) # callbacks=[batch_history]
        print "before emptying"
        tm.sleep(7)

        X_train = None
        y_train = None

        print "after emptying"

    #   last_loss = epoch_history.history['val_loss'][-1]
    #   last_acc = epoch_history.history['val_acc'][-1]
    # self.batch_history = batch_history
    # self.epoch_history = epoch_history

  def preprocess_data(self, X, y):
    print "Im inside"
    X -= np.mean(X,axis=0)
    X_flip = np.zeros(X.shape)
    for i in xrange(X.shape[0]):
      for j in xrange(X.shape[1]):
        X_flip[i][j] = np.fliplr(X[i][j])
    X = np.concatenate((X, X_flip))

    tm.sleep(7)

    y = np.concatenate((y,y))
    y = np_utils.to_categorical(y)

    print "Im leaving"
    return X, y

def make_predictions(hyperparams):
  maker = ModelMaker(hyperparams)
  maker.create_model()
  maker.compile_model()
  maker.fit_data()

  X_eleven = np.load(path+'/data/chunks/aX_chunk11of12.npy').astype('float32')
  X_twelve = np.load(path+'/data/chunks/aX_chunk12of12.npy').astype('float32')
  print "Predicting now ..."
  eleven_preds = maker.model.predict(X_eleven, batch_size=batch_size)
  twelve_preds = maker.model.predict(X_twelve, batch_size=batch_size)
  predictions = np.concatenate((eleven_preds, twelve_preds))
  print predictions.shape
  return predictions

def display_results(final_predictions):
  pre = np.load(path+'/data/chunks/ay_chunk11of12.npy')
  post = np.load(path+'/data/chunks/ay_chunk12of12.npy')
  y_test = np.concatenate((pre,post))
  test_len = len(y_test)

  y_hat = np.argmax(final_predictions,axis=1)
  acc = np.sum(y_hat == y_test)
  print 'Final accuracy is', round(acc, 3)

hyperparams_list = generate_hyperparams(n_trials)
for trial in range(n_trials):
  print '\n ------------- RUNNING CROSS VALIDATION TRIAL', trial+1, '-------------'
  predictions = make_predictions(hyperparams_list[trial])
  display_results(predictions)
print "\n Scale Net is done."