# -*- coding: utf-8 -*-
# =============================================================================
# THE FOLLOWING CNN APPROACH 
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
import tensorflow as tf
import tensorflow.keras as keras
import scipy.stats
from sklearn.metrics import mean_squared_error
import os
# My Modules
from S3postproc import rescale_between

# =============================================================================
# BEGIN
# =============================================================================

#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files\S3A_2018-05-25 14_54_00__2018-05-25 02_16_36.npz'.replace('\\','\\')
filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr\S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz'.replace('\\','\\')
models_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\1DCNN\model_files'.replace('\\','\\') 	

# load npz files. encoding argument is used only if npz files have been
# saved using py2.x and are loaded by py3.x
dat = np.load(filename, encoding='latin1', allow_pickle=True)#encoding='latin1')

# Retrieve dictionary
dat = dat['arr_0'].item()
# Keep distance in variable
distance = dat['Distance']
del dat['Distance']

# count how many SST's are there
k = 0
for key in dat.keys():
    if 'SST' in key:
        k = k + 1

matrix = np.ma.zeros((distance.size, k), fill_value=np.nan)
#matrix = np.zeros((distance.size, k))
matrix_labels = [] # Name of variable

ub = 1
lb = -1

# Create feature matrix
i = 0
for key in dat.keys():
    if (type(dat[key]) is np.ma.masked_array): # Check if value of dictionary is np array
        if 'SST' in key:
            matrix[:, i] = np.squeeze(dat[key])
            matrix_labels.append(key)
            i = i + 1
    elif type(dat[key]) is np.ndarray:
        if 'SST' in key:
            matrix[:, i] = np.squeeze(rescale_between(dat[key].data, ub, lb))
            matrix_labels.append(key)
            i = i + 1
    # Assign SSHA to variable
    if 'SSHA' in key:
        label = dat[key]
# =============================================================================
# MISSING VALUES IMPUTATION            
# =============================================================================
# Convert masked array to ndarray
matrix = matrix.data
#matrix = np.ma.fix_invalid(matrix)
varmin = np.nanmin(matrix, axis=0, keepdims=True)
varmax = np.nanmax(matrix, axis=0, keepdims=True)
matrix = lb + (matrix - varmin)/(varmax - varmin) * (ub - lb)

if False:
    # Replace NANs of label (make them equal to 0)
    label_nan_mask = label.mask
    label.data[label_nan_mask] = 0
    label = label.data
    
    # Replace NANs of matrix
    matrix[np.isnan(matrix)] = 0

else:
    # Import Pandas
    import pandas as pd
    label_nan_mask = label.mask
    # Replace NANs of label (make them equal to 0)
    label_size = label.size # number of label elements
    label = pd.DataFrame(label.data)
    # Interpolate the NAN values inside the dataset
    label = label.interpolate(method='akima', limit=500, limit_direction='both', axis=0)
    # Interpolate (actually extrapolate) the values at the edges
    label = label.interpolate(method='linear', limit=500, limit_direction='both', axis=0)
    label = np.asarray(label).reshape(label_size)
    
    matrix_shape = matrix.shape # shape of matrix
    # Replace NANs of matrix
    matrix = pd.DataFrame(matrix)
    # Interpolate the NAN values inside the dataset
    matrix = matrix.interpolate(method='akima', limit=1500, limit_direction='both', axis=0)
    # Interpolate (actually extrapolate) the values at the edges
    matrix= matrix.interpolate(method='linear', limit=1500, limit_direction='both', axis=0)
    matrix = np.asarray(matrix).reshape(matrix_shape)
    

plt.figure(figsize=(15,8))
plt.plot(distance, label)
plt.xlabel('Distance [m]', fontsize=18)
plt.ylabel('Normalized Variables', fontsize=18)
plt.legend(['SSHA [m]'], fontsize=18)
plt.title('SST versions vs SSHA', fontsize=23)
i = 0
try:
    for key in matrix_labels:
        plt.plot(distance, matrix[:,i])
        i = i + 1
except:
    pass
    
del ub, lb, filename, dat, i, k#, varmax, varmin

# =============================================================================
# CNN INPUTS/LABEL SETUP
# =============================================================================

label = np.expand_dims(np.expand_dims(label, axis=1), axis=2)
matrix = np.expand_dims(matrix, axis=1)

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

def my_huber_loss(y_true, y_predict):
    pass
    return None
# Erase
# =============================================================================
# SETUP NETWORK
# =============================================================================

# Create array with indices
idx = np.arange(label.squeeze().size)

# SPLIT train-test randomly
x_train, x_test, label_train, label_test, idx_train, idx_test, label_nan_mask_train, label_nan_mask_test = sk.train_test_split(matrix, label, idx, label_nan_mask, test_size=0.30)

# create model
model = keras.Sequential()

# 1st pack of Conv/Pooling
# Add 1st layer (Convolution)
model.add(keras.layers.Conv1D(filters=4*16, # number of units
                                 kernel_size=1, # window size
                                 input_shape=(1, matrix.shape[2])))#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

#model.add(keras.layers.Dropout(0.5))

# 2nd pack of Conv/Pooling
model.add(keras.layers.Conv1D(filters=3*16, # number of units
                                 kernel_size=1)) # window size#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# 3rd pack of Conv/Pooling
model.add(keras.layers.Conv1D(filters=2*16, # number of units
                                 kernel_size=1)) # window size#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# 4th pack of Conv/Pooling
model.add(keras.layers.Conv1D(filters=16, # number of units
                                 kernel_size=1)) # window size#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# 4th pack of Conv/Pooling
model.add(keras.layers.Conv1D(filters=8, # number of units
                                 kernel_size=1)) # window size#, activation='relu')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add 2nd layer (Pooling)
model.add(keras.layers.MaxPooling1D(pool_size=1,
                                    strides=1))
#                                    padding='valid'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add de-noise layer
#model.add(keras.layers.Dropout(0.5))

# Add 4th layer (Fully Connected)
model.add(keras.layers.Dense(units=1, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Compile model
model.compile(loss=charbonnier,#tf.losses.huber_loss,#my_loss,#'mse', # Loss function
              optimizer='rmsprop')
              # metrics=['mean_squared_error']) # Optimizer

# Create Callback variables
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = keras.callbacks.ModelCheckpoint(os.path.join(models_path, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)

## non-nan values
#mask_useful = ~(x_test.mask).any(axis=2)

# Train model
history = model.fit(x=x_train, y=label_train,
          batch_size=5,
          epochs=200,
          verbose=1, # progress bar
          validation_split=0.2,
          shuffle=True,
          callbacks=[es, mc]) # fraction of data as validation

print(history.history.keys())
# Compute errors
#loss = model.evaluate(x=x_test, y=label_test,
#                           verbose=1, # progress bar
#                           batch_size=20)

## non-nan values
#mask_predict = ~(matrix.mask).any(axis=2)


saved_model = keras.models.load_model(os.path.join(models_path, 'best_model.h5'), custom_objects={'charbonnier': charbonnier})

## Predict on "unseen" data
y_hat = model.predict(matrix, verbose=1)
y_hat = np.squeeze(y_hat)

yhat2 = model.predict(x_test, verbose=1)
yhat2 = yhat2.squeeze()
# Predict on unseen data
#y_hat = model.predict(x_test)
#y_hat = np.squeeze(y_hat)

idx_test = np.sort(idx_test)
# Plot prediction and label
font = {'size' : 18}
plt.rc('font', **font)
fig, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
ax1.scatter(distance[idx_test], label[:,0,0][idx_test])
ax1.scatter(distance[idx_test], y_hat[idx_test],c='red', s=3)
ax1.set_ylabel('SSHA [m]', fontsize=18)
ax1.set_xlabel('Distance [m]', fontsize=18)
ax1.legend(['Test_label', 'Test_predicted'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('Loss [m]', fontsize=18)
ax2.set_xlabel('Epoch', fontsize=18)
ax2.set_xlim(0, len(history.history['loss']))
ax2.legend(['Train', 'Validation'], loc='upper left')

# =============================================================================
# REGRESSIONS ON LABEL (TEST) AND PREDICTED (TEST)
# =============================================================================
res_thsen= scipy.stats.theilslopes(label_test.squeeze()[~label_nan_mask_test], yhat2[~label_nan_mask_test], 0.95)
res_ols = scipy.stats.linregress(label_test.squeeze()[~label_nan_mask_test], yhat2[~label_nan_mask_test])
RMSE = np.sqrt(mean_squared_error(label_test.squeeze()[~label_nan_mask_test], yhat2[~label_nan_mask_test]))

plt.figure(figsize=(10,7))
plt.scatter(label_test.squeeze()[~label_nan_mask_test], yhat2[~label_nan_mask_test])
# Theil-Sen regression
plt.plot(label_test.squeeze(), res_thsen[0]*label_test.squeeze() + res_thsen[1], 'r-')
# Linear Least Squares regression
plt.plot(label_test.squeeze(), res_ols[0]*label_test.squeeze() + res_ols[1], '#59ff00', linewidth=2.5)
plt.legend(['Theil-Sen', 'OLS'], loc='lower right')
plt.axis('equal')
plt.xlabel('Label SSHA [m]')
plt.ylabel('Predicted SSHA [m]')
#plt.xlim([-1, 1])
plt.text(0.2, -0.1, 'Theis-Sen\n'+'y = {0:.4f} x + {1:.4f}\nRMSE = {2:.4f}\nOLS\ny = {3:.4f} x + {4:.4f}'.format(res_thsen[0], res_thsen[1], RMSE,res_ols[0],res_ols[1]), fontsize=13,
        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

train_loss = saved_model.evaluate(x_train, label_train, verbose=0)
test_loss = saved_model.evaluate(x_test, label_test, verbose=0)
print('Train loss: {0:.4f} [m]\nTest loss: {1:.4f} [m]'.format(train_loss, test_loss))

#json_string = model.to_json()
#with open(r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\1DCNN\cnn_13.json'.replace('\\','\\'), 'w') as json_file:  
#    json_file.write(json_string)