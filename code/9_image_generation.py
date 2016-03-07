from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D

img_width, img_height = 96, 96
train_level = 0

# this will contain our generated images
input_img = K.placeholder((1, 3, img_width, img_height))

# build the VGG16 network with our input_img as input
first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
first_layer.input = input_img

#load pre-trained model

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

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# Load the weights from our dropbox folder (about 0.5 GB worth) --------------------------
weights_path = 'MikeDerekNet.h5' #will eventually be our new pre-trained weights

f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')


from keras import backend as K

output_index = 0 #whatever class we want to target

layer_output = model.layers[-2].get_output()
loss = K.mean(layer_output[:,output_index])

# compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
# run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

img = input_img_data[0]
img = deprocess_image(img)
imsave('class_'+str(output_index)+'.png',img)
