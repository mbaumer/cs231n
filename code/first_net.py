'''
Transfer learning from VGG16
See https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3 for more info
'''
import h5py
import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D

weights_path = '/data/vgg16_weights.h5'
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

# In[6]:

for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')




# In[7]:

for layer in model.layers:
    layer.trainable = False


# In[8]:

# Add our own architecture --------------------------

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# model.add(Dense(256))
# model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation('softmax'))



# In[ ]:

model.layers[-1].trainable


# In[9]:

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils, generic_utils
X = np.load('../data/X.npy').astype('float32')
y = np.load('../data/Y.npy').astype('float32')
X -= np.mean(X,axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print y_train[0:5]
y_train, y_test = [np_utils.to_categorical(x) for x in (y_train, y_test)]
print y_train[0:5,:]


# In[10]:


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
#model.compile(loss='categorical_crossentropy', optimizer=Adam)

# I usually use batch_size = 128 during intial runs since it's faster
# Then to get better results, I lower it to 32
# I also usually start with 3 epochs and raise that number as far up as it will go
# Usually, that is only ~20 epochs
model.fit(X_train, y_train, batch_size=16, nb_epoch=10, verbose=1)



# In[11]:

predictions = model.predict(X_train,batch_size=16,verbose=1)


# In[13]:

test_predictions = model.predict(X_test,batch_size=16,verbose=1)


# In[14]:

np.sum(np.argmax(predictions,axis=1) == np.argmax(y_train,axis=1))

