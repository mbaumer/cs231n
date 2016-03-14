import numpy as np
import time as tm
from sklearn.utils import shuffle

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


'''
X[8:14, 1:2, 0:3, 0:3]

array([[[[ 230.,  186.,  182.],
         [ 217.,  205.,  180.],
         [ 222.,  183.,  173.]]],


       [[[ 246.,  248.,  249.],
         [ 250.,  248.,  247.],
         [ 252.,  247.,  247.]]],


       [[[ 252.,  252.,  252.],
         [ 252.,  252.,  252.],
         [ 252.,  252.,  252.]]],


       [[[  79.,  194.,   72.],
         [  99.,  172.,   73.],
         [  93.,  198.,   66.]]],


       [[[ 202.,  212.,  193.],
         [ 174.,  196.,  204.],
         [ 218.,  199.,  230.]]],


       [[[ 162.,  224.,  253.],
         [ 111.,  160.,  124.],
         [ 126.,  212.,  213.]]]], dtype=float32)

array([[[[  38.,   48.,   39.],
         [  35.,   36.,   37.],
         [  43.,   27.,   38.]]],

       [[[ 202.,  207.,  203.],
         [ 207.,  199.,  202.],
         [ 203.,  199.,  211.]]],

       [[[ 153.,  151.,  183.],
         [ 159.,  143.,  187.],
         [ 147.,  126.,  141.]]],

       [[[ 188.,  209.,  207.],
         [ 176.,  150.,  168.],
         [ 177.,  191.,  157.]]],

       [[[  68.,  112.,  138.],
         [  51.,  100.,  132.],
         [   7.,   27.,  112.]]],

       [[[ 123.,  129.,  128.],
         [  92.,   89.,  104.],
         [  87.,   90.,  101.]]]], dtype=float32)



from keras import backend as K
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D


for i in xrange(num_chunks):
  chunk_data = load_chunk(i)
  X_chunk.set_value(chunk_data)
  for minibatch in xrange(num_minibatches):
      train_batch(minibatch) # train_batch should be using X_chunk

from keras.utils import np_utils, generic_utils, io_utils

get_h5 = io_utils.HDF5Matrix
#API found here https://github.com/fchollet/keras/blob/master/keras/utils/io_utils.py
# Assumes features and labels are in HDF5 datasets named 'features' and 'targets':
# Requires that we (a) split the test/train and (b) shuffle the data, before storing in h5 files
def load_data(datapath, start, chunk_size):
  X_train = get_h5('/data/X_artists_train.h5', 'features', start, start+chunk_size, normalizer=normalize_data)
  y_train = get_h5('/data/y_artists_train.h5', 'targets', start, start+chunk_size)
  return X_train, y_train

# Image generator that yields ~10k samples at a time.
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32): # these are chunks of 32 samples
            loss = model.train(X_batch, Y_batch)

# Alternatively, without data augmentation / normalization:
for e in range(nb_epoch):
    print("epoch %d" % e)
    for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
        model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
'''