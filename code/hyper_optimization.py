from __future__ import division
'''
Transfer learning from VGG16
See https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 for more info
'''
import h5py
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
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

weights_path = '/data/vgg16_weights.h5'
training_input = '/data/X.npy'
training_output = '/data/Y.npy'
img_width, img_height = 128, 128

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

for layer in model.layers:
	layer.trainable = False

# Add our own architecture --------------------------

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256,name='dense_1'))
model.add(Activation('relu'))
model.add(Dropout(0.5,name='dropout_1'))
# model.add(Dense(256))
# model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))



X = np.load(training_input).astype('float32')
y = np.load(training_output).astype('float32')
X -= np.mean(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

class CrossValidator(object):

	def __init__(self):
		self.batch_histories = []
		self.epoch_histories = []
		self.best_model = None
		self.best_val_loss = 1e9

	def run(self,n_trials):
		for i in range(n_trials):
			learning_rate = 10**np.random.uniform(-5,-2,1)
			#dropout_prob = np.random.uniform(.5,.95,1)
			#dense_regularization = 10**np.random.uniform(-6,-2,1)
			#model.layers['dropout_1'].p = dropout_prob[0]
			#model.layers['dense_1'].W_regularizer = l2(dense_regularization[0])
			print 'running crossval trial', i, 'learning rate is', learning_rate
			adam = Adam(lr=learning_rate[0], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
			model.compile(loss='categorical_crossentropy', optimizer=adam)
			print "Model has compiled."

			batch_history = LossHistory()
			epoch_history = model.fit(X_train, y_train, batch_size=32, nb_epoch=3, verbose=1, show_accuracy=True, callbacks=[batch_history], validation_split=0.2)
			print 'last epoch val loss for this iteration is', epoch_history.history['val_loss'][-1], 'current best is', self.best_val_loss
			print 'last val acc is', epoch_history.history['val_acc'][-1]
			if i == 0:
				print 'first iteration; saving model'
				self.best_model = copy(model)
				self.best_val_loss = epoch_history.history['val_loss'][-1]
			elif epoch_history.history['val_loss'][-1] < self.best_val_loss:
				print 'I think this current model is better: Im saving it.'
				self.best_model = copy(model)
				self.best_val_loss = epoch_history.history['val_loss'][-1]

			self.batch_histories.append(batch_history.losses)
			self.epoch_histories.append(epoch_history.history['val_loss'])

			# print "Train Accuracy"
			# train_predictions = model.predict(X_train, batch_size=32, verbose=1)
			# print np.sum(np.argmax(train_predictions,axis=1) == np.argmax(y_train,axis=1))/X_train.shape[0]

			# print "Val Accuracy"
			# val_predictions = model.predict(X_val,batch_size=32,verbose=1)
			# print np.sum(np.argmax(val_predictions,axis=1) == np.argmax(y_val,axis=1))/X_val.shape[0]

	def plot(self):
		fig1,ax1 = plt.figure()
		ax1.set_xlabel('Batch Number')
		ax1.set_ylabel('Training Loss')
		for history in self.batch_histories:
			ax1.plot(history)
		fig1.savefig('training_losses.png')

		fig2,ax2 = plt.figure()
		ax2.set_xlabel('Epoch Number')
		ax2.set_ylabel('Validation Loss')
		for history in self.epoch_histories:
			ax2.plot(history)
		fig2.savefig('validation_losses.png')

solver = CrossValidator()
solver.run(5)
print 'best model has learning rate of', solver.best_model.optimizer.lr
train_predictions = solver.best_model.predict(X_train, batch_size=32, verbose=1)
print 'training accuracy is', np.sum(np.argmax(train_predictions,axis=1) == np.argmax(y_train,axis=1))/X_train.shape[0]
# val_predictions = solver.best_model.predict(X_val,batch_size=32,verbose=1)
# print 'validation accuracy is', np.sum(np.argmax(val_predictions,axis=1) == np.argmax(y_val,axis=1))/X_val.shape[0]
test_predictions = solver.best_model.predict(X_test,batch_size=32,verbose=1)
print 'test accuracy is', np.sum(np.argmax(test_predictions,axis=1) == np.argmax(y_test,axis=1))/X_test.shape[0]
solver.plot()
print 'The loss curves pictures are in current directory, named as training_losses.png and validation_losses.png'

