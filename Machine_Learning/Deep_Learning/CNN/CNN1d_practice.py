# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# 1D Convolution (take N samples of the sequence as training and the rest as testing)
# N can be chosen either at random either non-random (E.g. the first N consecutive samples of the sequence)
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk

# =============================================================================
# BEGIN
# =============================================================================

seed = 7
np.random.seed(seed)

# GENERATE DUMMY DATA
spatial_res = 330 # sampling distance
max_distance = 800000 # [m]
num_sample = max_distance//spatial_res # number of samples

T = 200000 # wavelength [m]
omega = 2*np.pi/T
d = np.linspace(0, max_distance, num=num_sample) # generate distance vector

x = np.sin(omega*d) + np.sin(omega/2.0*d + 100000) + np.random.normal(size=num_sample)*0.02
x = x.astype('float32')
x = np.expand_dims(x, axis=1)

x2 = 0.2*np.sin(omega*d + 100000) + np.random.normal(size=num_sample)*0.02
x2 = x2.astype('float32')
x2 = np.expand_dims(x2, axis=1)

x = np.concatenate((x, x2), axis=1)

# SPLIT train-test randomly
x_train, x_test, label_train, label_test = sk.train_test_split(x, x, test_size=0.30)
# SPLIT train-test by taking consecutive elements
#x_train = x[:2000]
#x_test = x[2000:]
#label_train = x[:2000]
#label_test = x[2000:]

# CHANGE DIMENSIONS
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
label_train = np.expand_dims(label_train, axis=2)
label_test = np.expand_dims(label_test, axis=2)

# BUILD THE CNN
# create model
model = keras.Sequential()

# Add 1st layer (Convolution)
model.add(keras.layers.Conv1D(filters=32, # number of units
                                 kernel_size=1, # window size
                                 input_shape=(2, 1)))#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 3rd layer (flatten)
#model.add(keras.layers.Flatten())

# Add 4th layer (Fully Connected)
model.add(keras.layers.Dense(units=1, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Compile model
model.compile(loss='mse', # Loss function
              optimizer='rmsprop',
              metrics=['mean_squared_error']) # Optimizer

# Train model
model.fit(x=x_train, y=label_train,
          batch_size=32,
          epochs=5,
          verbose=1, # progress bar
          validation_data=(x_test, label_test))

# Compute errors
loss, mse = model.evaluate(x=x_test, y=label_test,
                           verbose=1, # progress bar
                           batch_size=128)

# Predict on "unseen" data
y_hat = model.predict(np.expand_dims(x, axis=2))
y_hat = np.squeeze(y_hat)

# PLOT
plt.figure(figsize=(9,6))
plt.plot(d, x[:,0], d, y_hat[:,0])
plt.legend(('original', 'prediction'))

plt.figure(figsize=(9,6))
plt.plot(d, x[:,1], d, y_hat[:,1])
plt.legend(('original', 'prediction'))

#%%
# =============================================================================
# 1D Convolution (take N samples of the sequence as training and the rest as testing)
# N can be chosen either at random either non-random (E.g. the first N samples of the sequence)
# =============================================================================
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import random
import sys

# =============================================================================
# MY LOSS FUNCTIONS
# =============================================================================
def my_loss(y_true, y_predict):
    # mask out nans
    idx = tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(y_true), tf.math.is_nan(y_predict)))
    # Compute MSE
    mse = keras.backend.mean((tf.boolean_mask(y_true, idx) - tf.boolean_mask(y_predict, idx))**2)
    
    return mse

def charbonnier(y_true, y_predict):
    epsilon = 1e-3
    error = y_true - y_predict
    p = keras.backend.sqrt(keras.backend.square(error) + keras.backend.square(epsilon))
    
    return keras.backend.mean(p)

# =============================================================================
# BEGIN
# =============================================================================
#
seed = 7607888747476562202
#seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print('Seed was: {0}\n'.format(seed))

# GENERATE DUMMY DATA
spatial_res = 330 # sampling distance
max_distance = 800000 # [m]
num_sample = max_distance//spatial_res # number of samples

T = 200000 # wavelength [m]
omega = 2*np.pi/T
d = np.linspace(0, max_distance, num=num_sample) # generate distance vector

# LABEL
label = np.expand_dims(np.expand_dims(np.sin(omega*d), axis=1), axis=2)

# INPUT
# Signal 1
x = np.sin(omega*d) + np.sin(omega/2.0*d + 100000) + np.random.normal(size=num_sample)*0.02
x = x.astype('float32')
x = np.expand_dims(np.expand_dims(x, axis=1), axis=2)
# Signal 2
x2 = 0.2*np.sin(omega*d + 100000) + np.random.normal(size=num_sample)*0.02
x2 = x2.astype('float32')
x2 = np.expand_dims(np.expand_dims(x2, axis=1), axis=2)
# Signal 3
x3 = np.sin(omega*d - np.linspace(0, 2*np.pi, num_sample)) # varying phase signal
x3 = x3.astype('float32')
x3 = np.expand_dims(np.expand_dims(x3, axis=1), axis=2)
# Signal 4
trend_phase1 = np.linspace(0, 2*np.pi, num_sample//2)
trend_phase2 = np.linspace(2*np.pi, 0, num_sample//2)
x4 = np.sin(omega*d - np.concatenate((trend_phase1, trend_phase2))) # varying phase signal
x4 = x4.astype('float32')
x4 = np.expand_dims(np.expand_dims(x4, axis=1), axis=2)
# Signal 5
x5 = np.sin(omega/8*d)
x5 = x5.astype('float32')
x5 = np.expand_dims(np.expand_dims(x5, axis=1), axis=2)
# Signal 6 (Brownian walk)
# Change standard deviation argument to change the Signal-to-Noise ratio
b_walk = np.random.normal(loc=0, # mean
                          scale=1/12, # standard deviation
                          size=num_sample)
b_walk = np.cumsum(b_walk)
b_walk = np.expand_dims(np.expand_dims(b_walk, axis=1), axis=2)
x6 = label + b_walk # signal + brown walk
# Signal 7 (New Brownian walk shifted)
# Change standard deviation argument to change the Signal-to-Noise ratio
b_walk2 = np.random.normal(loc=0, # mean
                          scale=1/5, # standard deviation
                          size=num_sample)
b_walk2 = np.cumsum(b_walk2)
b_walk = np.expand_dims(np.expand_dims(b_walk2, axis=1), axis=2)
x7 = label + b_walk # signal + brown walk
x7 = np.roll(x7, shift=200,axis=0) # shift vector
# Constant signal equal to zeros
x8 = np.zeros((num_sample))
x8 = x8.astype('float32')
x8 = np.expand_dims(np.expand_dims(x8, axis=1), axis=2)

# Feature matrix
x = np.concatenate((x, x2, x3, x4, x5, x6, x7, x8), axis=2)

# =============================================================================
# MAKE x A MASKED NDARRAY
# =============================================================================
my_mask = np.zeros(x.shape, dtype='bool')
#my_mask[:, 0, 7] = np.ones((num_sample), dtype='bool')
my_mask[:, 0, 7][0::25] = True
#np.ones((num_sample), dtype='bool')
x = np.ma.array(x, mask=my_mask)
x.data[:, 0, 7][0::25] = 0#np.nan#-9999

# SPLIT train-test randomly
x_train, x_test, label_train, label_test = sk.train_test_split(x, label, test_size=0.20)

# SPLIT train-test by taking consecutive elements
#n_elem = 2000
#x_train = x[:n_elem]
#x_test = x[n_elem:]
#label_train = x[:n_elem]
#label_test = x[n_elem:]

# CHANGE DIMENSIONS
#x_train = np.expand_dims(x_train, axis=2)
#x_test = np.expand_dims(x_test, axis=2)
#label_train = np.expand_dims(label_train, axis=2)
#label_test = np.expand_dims(label_test, axis=2)

# BUILD THE CNN
# create model
model = keras.Sequential()

# Add 1st layer (Convolution)
model.add(keras.layers.Conv1D(filters=32, # number of units
                                 kernel_size=1, # window size
                                 input_shape=(1, x.shape[2])))#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 3rd layer (flatten)
#model.add(keras.layers.Flatten())

# Add de-noise layer
#model.add(keras.layers.Dropout(0.2))

# Add 4th layer (Fully Connected)
model.add(keras.layers.Dense(units=1, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Compile model
model.compile(loss=charbonnier,#my_loss,#'mse', # Loss function
              optimizer='rmsprop',
              metrics=['mean_squared_error']) # Optimizer

# non-nan values
mask_useful = ~(x_test.mask).any(axis=2)

# Train model
history = model.fit(x=x_train, y=label_train,
          batch_size=20,
          epochs=40,
          verbose=1, # progress bar
          validation_split=0.2)
          #validation_data=(x_test, label_test)
print(history.history.keys())
# Compute errors
loss = model.evaluate(x=x_test[mask_useful.squeeze(),:,:], y=label_test[mask_useful.squeeze(),:,:],
                           verbose=1, # progress bar
                           batch_size=20)

# non-nan values
mask_predict = ~(x.mask).any(axis=2)

# Predict on "unseen" data
y_hat = model.predict(x[mask_predict.squeeze(),:,:])
y_hat = np.squeeze(y_hat)

# PLOT
fig, [ax1, ax2, ax3] = plt.subplots(3,1, sharex=False, figsize=(9,8))

ax1.plot(d[mask_predict.squeeze()],label[mask_predict.squeeze(),0,0],
         d,x[:,0,0],
         d, x[:,0,1],
         d, x[:,0,2],
         d, x[:,0,3],
         d, x[:,0,4],
         d, x[:,0,5],
         d, x[:,0,6],
         d, x[:,0,7],
         d[mask_predict.squeeze()], y_hat)
ax1.legend(('LABEL',
            'Signal 1',
            'Signal 2',
            'Signal 3',
            'Signal 4',
            'Signal 5',
            '+Brownian walk',
            '+Brownian walk 2 shifted',
            'Zeros',
            'Prediction'))
ax1.set_xlabel('Distance [m]')
ax1.set_ylabel('Signal')

ax2.plot(d[mask_predict.squeeze()],label[mask_predict.squeeze(),0,0], d[mask_predict.squeeze()], y_hat)
ax2.legend(('LABEL', 'Prediction'))
ax2.set_xlabel('Distance [m]')
ax2.set_ylabel('Signal')

ax3.plot(history.history['loss'])
ax3.plot(history.history['val_loss'])
ax3.set_ylabel('Loss')
ax3.set_xlabel('Epoch')
ax3.set_xlim(0, len(history.history['loss']))
ax3.legend(['Train', 'Test'], loc='upper left')

#%%

# =============================================================================
# 1D Convolution (take N samples of the sequence as training and the rest as testing)
# N can be chosen either at random either non-random (E.g. the first N samples of the sequence)
# =============================================================================
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import random
import sys

# =============================================================================
# MY LOSS FUNCTIONS
# =============================================================================
def my_loss(y_true, y_predict):
    # mask out nans
    idx = tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(y_true), tf.math.is_nan(y_predict)))
    # Compute MSE
    mse = keras.backend.mean((tf.boolean_mask(y_true, idx) - tf.boolean_mask(y_predict, idx))**2)
    
    return mse

def charbonnier(y_true, y_predict):
    epsilon = 1e-3
    error = y_true - y_predict
    p = keras.backend.sqrt(keras.backend.square(error) + keras.backend.square(epsilon))
    
    return keras.backend.mean(p)

# =============================================================================
# BEGIN
# =============================================================================
#
seed = 7607888747476562202
#seed = random.randrange(sys.maxsize)
rng = random.Random(seed)
print('Seed was: {0}\n'.format(seed))

# GENERATE DUMMY DATA
spatial_res = 330 # sampling distance
max_distance = 800000 # [m]
num_sample = max_distance//spatial_res # number of samples

T = 200000 # wavelength [m]
omega = 2*np.pi/T
d = np.linspace(0, max_distance, num=num_sample) # generate distance vector

# LABEL
label = np.expand_dims(np.expand_dims(np.sin(omega*d), axis=1), axis=2)

# INPUT
# Signal 1
x = np.sin(omega*d) + np.sin(omega/2.0*d + 100000) + np.random.normal(size=num_sample)*0.02
x = x.astype('float32')
x = np.expand_dims(np.expand_dims(x, axis=1), axis=2)
# Signal 2
x2 = 0.2*np.sin(omega*d + 100000) + np.random.normal(size=num_sample)*0.02
x2 = x2.astype('float32')
x2 = np.expand_dims(np.expand_dims(x2, axis=1), axis=2)
# Signal 3
x3 = np.sin(omega*d - np.linspace(0, 2*np.pi, num_sample)) # varying phase signal
x3 = x3.astype('float32')
x3 = np.expand_dims(np.expand_dims(x3, axis=1), axis=2)
# Signal 4
trend_phase1 = np.linspace(0, 2*np.pi, num_sample//2)
trend_phase2 = np.linspace(2*np.pi, 0, num_sample//2)
x4 = np.sin(omega*d - np.concatenate((trend_phase1, trend_phase2))) # varying phase signal
x4 = x4.astype('float32')
x4 = np.expand_dims(np.expand_dims(x4, axis=1), axis=2)
# Signal 5
x5 = np.sin(omega/8*d)
x5 = x5.astype('float32')
x5 = np.expand_dims(np.expand_dims(x5, axis=1), axis=2)
# Signal 6 (Brownian walk)
# Change standard deviation argument to change the Signal-to-Noise ratio
b_walk = np.random.normal(loc=0, # mean
                          scale=1/12, # standard deviation
                          size=num_sample)
b_walk = np.cumsum(b_walk)
b_walk = np.expand_dims(np.expand_dims(b_walk, axis=1), axis=2)
x6 = label + b_walk # signal + brown walk
# Signal 7 (New Brownian walk shifted)
# Change standard deviation argument to change the Signal-to-Noise ratio
b_walk2 = np.random.normal(loc=0, # mean
                          scale=1/5, # standard deviation
                          size=num_sample)
b_walk2 = np.cumsum(b_walk2)
b_walk = np.expand_dims(np.expand_dims(b_walk2, axis=1), axis=2)
x7 = label + b_walk # signal + brown walk
x7 = np.roll(x7, shift=200,axis=0) # shift vector
# Constant signal equal to zeros
x8 = np.zeros((num_sample))
x8 = x8.astype('float32')
x8 = np.expand_dims(np.expand_dims(x8, axis=1), axis=2)

# Feature matrix
x = np.concatenate((x, x2, x3, x4, x5, x6, x7, x8), axis=2)

# =============================================================================
# MAKE x A MASKED NDARRAY
# =============================================================================
my_mask = np.zeros(x.shape, dtype='bool')
#my_mask[:, 0, 7] = np.ones((num_sample), dtype='bool')
my_mask[:, 0, 7][0::25] = True
#np.ones((num_sample), dtype='bool')
x = np.ma.array(x, mask=my_mask)
x.data[:, 0, 7][0::25] = np.nan#-9999

# SPLIT train-test randomly
x_train, x_test, label_train, label_test = sk.train_test_split(x, label, test_size=0.20)

# SPLIT train-test by taking consecutive elements
#n_elem = 2000
#x_train = x[:n_elem]
#x_test = x[n_elem:]
#label_train = x[:n_elem]
#label_test = x[n_elem:]

# CHANGE DIMENSIONS
#x_train = np.expand_dims(x_train, axis=2)
#x_test = np.expand_dims(x_test, axis=2)
#label_train = np.expand_dims(label_train, axis=2)
#label_test = np.expand_dims(label_test, axis=2)

# BUILD THE CNN
# create model
model = keras.Sequential()


# Add 1st layer (Convolution)
model.add(keras.layers.Conv1D(filters=32, # number of units
                                 kernel_size=1, # window size
                                 input_shape=(1, x.shape[2])))# shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Add 3rd layer (flatten)
#model.add(keras.layers.Flatten())

# Add de-noise layer
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Masking(mask_value=np.nan))
# Add 4th layer (Fully Connected)
model.add(keras.layers.Dense(units=1, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.6)) # activation function

# Compile model
model.compile(loss='mse',#my_loss,#'mse', # Loss function
              optimizer='rmsprop',
              metrics=['mean_squared_error']) # Optimizer

# non-nan values
mask_useful = ~(x_test.mask).any(axis=2)

# Train model
model.fit(x=x_train, y=label_train,
          batch_size=32,
          epochs=7,
          verbose=1, # progress bar
          validation_split=0.2)
          #validation_data=(x_test, label_test)

# Compute errors
loss = model.evaluate(x=x_test, y=label_test,
                           verbose=1, # progress bar
                           batch_size=32)

# non-nan values
mask_predict = ~(x.mask).any(axis=2)

# Predict on "unseen" data
y_hat = model.predict(x)
y_hat = np.squeeze(y_hat)

# PLOT
fig, [ax1, ax2] = plt.subplots(2,1, sharex=True, figsize=(9,8))

ax1.plot(d,label.squeeze(),
         d,x[:,0,0],
         d, x[:,0,1],
         d, x[:,0,2],
         d, x[:,0,3],
         d, x[:,0,4],
         d, x[:,0,5],
         d, x[:,0,6],
         d, x[:,0,7],
         d[~my_mask[:, 0, 7].squeeze()], y_hat[~my_mask[:, 0, 7].squeeze()])
ax1.legend(('LABEL',
            'Signal 1',
            'Signal 2',
            'Signal 3',
            'Signal 4',
            'Signal 5',
            '+Brownian walk',
            '+Brownian walk 2 shifted',
            'Zeros',
            'Prediction'))
ax1.set_xlabel('Distance [m]')
ax1.set_ylabel('Signal')

ax2.plot(d,label.squeeze(), d[~my_mask[:, 0, 7].squeeze()], y_hat[~my_mask[:, 0, 7].squeeze()])
ax2.legend(('LABEL', 'Prediction'))
ax2.set_xlabel('Distance [m]')
ax2.set_ylabel('Signal')
