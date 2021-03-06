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

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

path_head = '/Users/mbaumer/Documents/CS231n/project/cs231n'
weights_path = path_head+'/data/vgg16_weights.h5'
training_input = path_head+'/data/X.npy'
training_output = path_head+'/data/Y.npy'
img_width, img_height = 128,128
epoch_count = 5

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
		self.best_model_params = {}

	def run(self, n_trials):
		for trial in range(n_trials):

			learning_rate = 10**np.random.uniform(-6,-3)
			reg_strength = 10**np.random.uniform(-7,0)
			dropout_prob = np.random.uniform(0.1,0.9)

			print 'Running crossval trial', trial+1, 'Learning rate is', learning_rate
			print 'reg_strength is', reg_strength, 'dropout_prob is', dropout_prob

			model = createModel()

			model.add(Flatten())
			# Note: Keras does automatic shape inference.
			model.add(Dense(4096,name='dense_1',init='he_normal',W_regularizer=l2(reg_strength)))
			model.add(BatchNormalization(momentum=0.9))
			model.add(Activation('relu'))
			model.add(Dropout(dropout_prob,name='dropout_1'))

			model.add(Dense(4096,name='dense_2',init='he_normal',W_regularizer=l2(reg_strength)))
			model.add(BatchNormalization(momentum=0.9))
			model.add(Activation('relu'))
			model.add(Dropout(dropout_prob,name='dropout_2'))

			model.add(Dense(1000,name='dense_3',init='he_normal',W_regularizer=l2(reg_strength)))
			model.add(BatchNormalization(momentum=0.9))
			model.add(Activation('relu'))
			model.add(Dropout(dropout_prob,name='dropout_3'))

			model.add(Dense(3))
			model.add(Activation('softmax'))

			adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

			initial_time = tm.time()
			model.compile(loss='categorical_crossentropy', optimizer=adam)
			checkpoint = tm.time() - initial_time
			print 'Compiled in %s seconds' % round(checkpoint, 3)

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
				self.best_model_params = {'learning_rate': learning_rate, 'reg_strength': reg_strength, 'dropout_prob': dropout_prob}

			elif epoch_history.history['val_loss'][-1] < self.best_val_loss:
				print 'I think this current model is better: Im saving it.'
				self.best_model = copy(model)
				self.best_val_loss = epoch_history.history['val_loss'][-1]
				self.best_model_params = {'learning_rate': learning_rate, 'reg_strength': reg_strength, 'dropout_prob': dropout_prob}

			self.batch_histories.append(batch_history.losses)
			self.epoch_histories.append(epoch_history.history['val_loss'])
			self.epoch_acc_histories.append(epoch_history.history['val_acc'])

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

solver = CrossValidator()
solver.run(1)
print 'best model has learning rate of', str(solver.best_model.best_model_params['learning_rate'])
print 'best model has reg strength of', str(solver.best_model.best_model_params['reg_strength'])
print 'best model has dropout prob of', str(solver.best_model.best_model_params['dropout_prob'])
train_predictions = solver.best_model.predict(X_train, batch_size=32, verbose=1)
print 'training accuracy is', np.sum(np.argmax(train_predictions,axis=1) == np.argmax(y_train,axis=1))/X_train.shape[0]
# val_predictions = solver.best_model.predict(X_val,batch_size=32,verbose=1)
# print 'validation accuracy is', np.sum(np.argmax(val_predictions,axis=1) == np.argmax(y_val,axis=1))/X_val.shape[0]
test_predictions = solver.best_model.predict(X_test,batch_size=32,verbose=1)
print 'test accuracy is', np.sum(np.argmax(test_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]
np.save('test_pred.npy',test_predictions)
np.save('train_pred.npy',train_predictions)
solver.plot()
print 'Loss curves images are in current directory, named as train_loss.png and val_loss.png'

