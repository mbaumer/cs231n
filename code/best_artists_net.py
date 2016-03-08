from __future__ import division
import h5py
import time as tm
import numpy as np
import pandas as pd
import pickle
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
from keras.utils import np_utils, generic_utils

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

env = 'remote'
if env == 'local':
  # path = '/Users/mbaumer/Documents/CS231n/project/cs231n'
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

train_level = 0
augment = False

if env == 'local':
  classes = 3
  rates = [7.4e-5, 1.2e-5, 4.4e-6]
  n_trials = 3
  epoch_count = 3
  img_width, img_height = 128, 128
  target_w, target_h = 96, 96
  X_train, X_test = X_train[:50,:,:,:], X_test[:50,:,:,:]
  y_train, y_test = y_train[:50,:], y_test[:50,:]
  batch_size = 20
elif env == 'remote':
  classes = 20
  n_trials = 1
  epoch_count = 3
  img_width, img_height = 256, 256
  target_w, target_h = 224, 224
  batch_size = 32

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

class ModelMaker(object):

  def __init__(self,hyperparams,train_level,train_src=None,val_data=None):
    self.learning_rate = hyperparams[0]
    self.reg_strength = hyperparams[1]
    self.dropout_prob = round(hyperparams[2], 2)
    self.batch_history = None
    self.epoch_history = None
    self.model = None
    self.train_level = train_level
    self.train_src = train_src
    self.val_data = val_data
    print 'Learning rate is', self.learning_rate
    print 'Reg_strength is', self.reg_strength
    print 'Dropout_prob is', self.dropout_prob

  def create_model(self):
    # train_level = 0: only train FC layers
    #         1: train FC layers and last 3 conv layers
    #         2: train FC layers and last 6 conv layers

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, target_h, target_w)))
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
      # print g.keys()
      weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
      # print type(model.layers[k])
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

  def compile_model(self):
    adam = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    initial_time = tm.time()
    self.model.compile(loss='categorical_crossentropy', optimizer=adam)
    checkpoint = tm.time() - initial_time
    print 'Compiled in %s seconds' % round(checkpoint, 3)

  def fit_data(self):
    batch_history = LossHistory()

    if self.train_src is not None:
      #epoch_history = self.model.fit_generator(self.train_src.flow(X_train2, y_train2, batch_size=batch_size),
      #              samples_per_epoch=self.train_src.epoch_size, nb_epoch=epoch_count, verbose=1, show_accuracy=True,
      #              callbacks=[batch_history],validation_data=self.val_data)
      print self.train_src[0].shape
      epoch_history = self.model.fit(self.train_src[0], self.train_src[1], batch_size=batch_size, nb_epoch=epoch_count, verbose=1, show_accuracy=True, callbacks=[batch_history], validation_data=self.val_data)
    else:
      epoch_history = self.model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epoch_count, verbose=1, show_accuracy=True, callbacks=[batch_history], validation_split=0.2)

    last_loss = epoch_history.history['val_loss'][-1]
    last_acc = epoch_history.history['val_acc'][-1]
    print 'Last validation loss for this iteration is', round(last_loss,4)
    print 'Last validation accuracy is', round(last_acc,4)
    self.model.save_weights(path+'/data/bestArtistWeights.h5',overwrite=True)
    self.batch_history = batch_history
    self.epoch_history = epoch_history

class CropGenerator(ImageDataGenerator):
  '''
  Generate minibatches with realtime data augmentation.
  Returns 10 random crops from the original image used during training phase
  '''
  def get_crops(self, image, target_height, target_width, Nrandoms=5, deterministic=False):
    full_height= image.shape[-2]
    full_width = image.shape[-1]
    acceptable_height = full_height - target_height
    acceptable_width = full_width - target_width
    half_h = target_height/2
    half_w = target_width/2
    midpoint_h = full_height/2
    midpoint_w = full_width/2

    crops = []
    if deterministic:
      top_left = [0,0] #yay easy!
      top_right = [0, full_width - target_width]
      bottom_left = [full_height - target_height, 0]
      bottom_right = [full_height - target_height, full_width - target_width]
      middle = [midpoint_h - half_h, midpoint_w - half_w]
      corners = [top_left, top_right, bottom_left, bottom_right, middle]
      for corner in corners:
        h = corner[0]
        w = corner[1]
        crop = image[:, h:h+target_height, w:w+target_width]
        crops.append(crop)
    else:
      for i in range(Nrandoms):
        h = np.random.randint(0,acceptable_height)
        w = np.random.randint(0,acceptable_width)
        crop = image[:, h:h+target_height, w:w+target_width]
        crops.append(crop)
    crops = np.array(crops)
    return crops

  def fit(self, X, mode, rounds, seed=42, target_h=target_h, target_w=target_w):
    X = np.copy(X)

    print len(X)
    print rounds
    self.epoch_size = len(X)*rounds*2

    if self.featurewise_center:
      self.mean = np.mean(X, axis=0)
      X -= self.mean
    # if self.featurewise_std_normalization:
    #   self.std = np.std(X, axis=0)
    #   X /= self.std

    # build an empty array of the appropriate size
    if augment:
      if mode == 'test': rounds=5
      aX = np.zeros((rounds*X.shape[0],X.shape[1],target_h,target_w))
      for i in range(X.shape[0]):
        if mode == 'train':
          #image, target_height, target_width, Nrandoms=5, deterministic=False
          imgs = self.get_crops(X[i], target_h, target_w, Nrandoms=rounds)
        else:
          imgs = self.get_crops(X[i], target_h, target_w, deterministic=True)
        aX[i*rounds:i*rounds+rounds,:,:,:] = imgs
      X = aX
    return X

def preprocess_data(data_in, mode):
  generator = CropGenerator(featurewise_center=True,
      featurewise_std_normalization=False, horizontal_flip=True)
  X = generator.fit(data_in,mode=mode,rounds=5)
  if mode == 'train':
    #return generator
    return X
  else:
    return X

class CrossValidator(object):

  def __init__(self):
    self.batch_histories = []
    self.epoch_histories = []
    self.epoch_acc_histories = []
    # self.best_model = None
    self.best_val_loss = 1e9
    self.best_model_params = {}

  def plot(self, trial):
    # trl = str(trial)
    trl = tria

    plt.figure()
    plt.xlabel('Batch Number')
    plt.ylabel('Training Loss')
    for history in self.batch_histories:
      plt.plot(history)
    plt.savefig('train_loss_'+trl+'.png')

    plt.figure()
    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Loss')
    for history in self.epoch_histories:
      plt.plot(history)
    plt.savefig('val_loss_'+trl+'.png')

    plt.figure()
    plt.xlabel('Epoch Number')
    plt.ylabel('Validation Accuracy')
    for history in self.epoch_acc_histories:
      plt.plot(history)
    plt.savefig('val_accuracy_'+trl+'.png')

    history_dict = {'batch_histories': self.batch_histories,
      'epoch_histories': self.epoch_histories,
      'epoch_acc_histories': self.epoch_acc_histories}
    pickle.dump(history_dict,open('history_dict'+trl+'.out','wb'))

  def update(self,maker,iteration):
    if iteration == 0:
      print 'First iteration; saving model'
      # self.best_model = copy(maker.model)
      self.best_val_loss = maker.epoch_history.history['val_loss'][-1]
      self.best_model_params = {'learning_rate': maker.learning_rate,
       'reg_strength': maker.reg_strength, 'dropout_prob': maker.dropout_prob}

    elif maker.epoch_history.history['val_loss'][-1] < self.best_val_loss:
      print 'I think this current model is better: Im saving it.'
      # self.best_model = copy(maker.model)
      self.best_val_loss = maker.epoch_history.history['val_loss'][-1]
      self.best_model_params = {'learning_rate': maker.learning_rate,
        'reg_strength': maker.reg_strength, 'dropout_prob': maker.dropout_prob}

    self.batch_histories.append(maker.batch_history.losses)
    self.epoch_histories.append(maker.epoch_history.history['val_loss'])
    self.epoch_acc_histories.append(maker.epoch_history.history['val_acc'])

def print_accuracy(predictions, y_test):
  print "Test Accuracy:"
  y_hat = np.argmax(predictions,axis=1)
  y_actual = np.argmax(y_test,axis=1)
  print np.sum(y_hat == y_actual)/X_test.shape[0]

def generate_hyperparams(n_trials):
  # learning_rates = 10**np.random.uniform(-4,-3,n_trials).astype('float32')
  # dropout_probs = np.random.uniform(0.3,0.5,n_trials).astype('float32')
  n_trials = 1
  learning_rates = [1e-4] #[0.001, 0.0009, 0.0007, 0.0005]
  reg_strengths = [8e-4]#10**np.random.uniform(-5,-4,n_trials).astype('float32')
  dropout_probs = [0.4]#[0.4, 0.4, 0.4, 0.4]
  return zip(learning_rates,reg_strengths,dropout_probs)

def build_ensembles(hyperparams_list):
  ensemble_results = []

  if augment:
    train_gen = preprocess_data(X_train2,mode='train')
    y_train_aug = np.repeat(y_train2,5,axis=0)
    train_data = (train_gen,y_train_aug)
    X_val_aug = preprocess_data(X_val,mode='test')
    y_val_aug = np.repeat(y_val,5,axis=0)
    val_data = (X_val_aug,y_val_aug)
    X_test_aug = preprocess_data(X_test,mode='test')
    y_test_aug = np.repeat(y_test,5,axis=0)
    print val_data[0].shape, val_data[1].shape
    print X_train2.shape, y_train2.shape
  else:
    train_data = None
    val_data = None

  solver = CrossValidator()
  # data_source = preprocess_data(X_train, mode='train')

  if augment:
    print X_test_aug.shape
    print y_test_aug.shape

  for trial in range(n_trials):
    print '  '
    print '------------- RUNNING CROSS VALIDATION TRIAL', trial+1, '-------------'
    maker = ModelMaker(hyperparams_list[trial],train_level,train_src=train_data,val_data=val_data)
    maker.create_model()
    maker.compile_model()
    maker.fit_data()

    solver.update(maker,trial)
    print X_test.shape
    print y_test.shape
    if augment:
      test_predictions = maker.model.predict(X_test_aug, batch_size=batch_size)
      print_accuracy(test_predictions, y_test_aug)
    else:
      train_predictions = maker.model.predict(X_train, batch_size=batch_size) 
      print_accuracy(train_predictions, y_train)
      test_predictions = maker.model.predict(X_test, batch_size=batch_size)
      print_accuracy(test_predictions, y_test)
      np.save('artists_train_predictions.npy',train_predictions)
      np.save('artists_test_predictions.npy'. test_predictions)
      np.save('artists_train_answers.npy',y_train)
      np.save('artists_test_answers.npy'. y_test)
    # solver.plot(trial)
    ensemble_results.append(test_predictions)

  solver.plot('Mar7')

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

  #if we're augmenting, need to split out val set by hand.
if augment:
  X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


hyperparams_list = generate_hyperparams(n_trials)
ensemble_results = build_ensembles(hyperparams_list)
final_predictions = vote_for_best(ensemble_results)
print 'Final accuracy is', np.sum(np.argmax(final_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]
print "Eighth net is done."