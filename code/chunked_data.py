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