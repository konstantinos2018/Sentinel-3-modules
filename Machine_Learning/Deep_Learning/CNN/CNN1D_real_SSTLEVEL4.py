# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import tensorflow.keras as keras
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import pdb
import scipy.stats
from sklearn.metrics import mean_squared_error
# My Modules
import ml_utilities

#global x_train
#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files\S3A_2018-05-25 14_54_00__2018-05-25 02_16_36.npz'.replace('\\','\\')
#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr\S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz'.replace('\\','\\')
models_path = r"H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\1DCNN\Proper\This_PC\SST_LEVEL4\SSHA_105_SST_LEVEL4_October".replace('\\','\\')
filespath = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\npz_files_sral_sstL4_1DCNN_real'.replace('\\','\\')
npz_files = os.listdir(filespath)
npz_files = [filename for filename in npz_files if '.npz' in filename]

# load npz files. encoding argument is used only if npz files have been
# saved using py2s.x and are loaded by py3.x
# Calculate Maximum distance vector size
d = []
fff = []
font = {'size' : 18}
plt.rc('font', **font)
for filename in npz_files:
    plt.close('all')
    matrix, distance,_ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, filename))
#    if distance.size < 2000:
#        continue
    # Imputate NaNs
    matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=False)
    
    label = np.array(matrix['SSHA_35'])
#    label = ml_utilities.matrix_min_max_rescale(label, 1, -1, axis=0)
    matrix = matrix.drop(columns=['SSHA_35', 'SSHA_71', 'SSHA_105'])
#    matrix = ml_utilities.matrix_min_max_rescale(matrix, 0.5, -0.5, axis=0)
    matrix = np.array(matrix)
    matrix = ml_utilities.my_standardizer(matrix, matrix) # standardize
    label = ml_utilities.my_standardizer(np.expand_dims(label,axis=1), np.expand_dims(label,axis=1)) # standardize
    matrix = matrix.squeeze()
    label = label.squeeze()
#    _, ccorr, _, _ = plt.xcorr(label, matrix, usevlines=True, maxlags=400, normed=True, lw=2)
#    ccorr_max = np.nanmax(np.abs(ccorr))
#    ccorr_min = ccorr.min()
#    print(np.isnan(ccorr))
#    if (ccorr_max < 0.5):
#        continue
    
#    fig = plt.figure(figsize=(15,10))
#    plt.plot(distance, label, distance, matrix)
#    plt.legend(['SSHA', 'SST'])
#    plt.title(filename[:-4], fontsize=23)
#    fig.savefig(os.path.join(r"H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SST_LEVEL4\trajectories".replace('\\','\\'), 'traj_'+filename +'.png'), dpi=300, bbox_inches='tight')
    
    d.append(len(distance))
    fff.append(filename)

    
#    if distance.size >= 1800:
#        d.append(len(distance))
#        fff.append(filename)
npz_files = fff
        


d = np.array(d)
d_max = d.max()
num_tracks = d.size
num_features = matrix.shape[0]# - 3
del d, distance
#
# Initialization
label = np.zeros(shape=(num_tracks, d_max), dtype=np.float32)
matrix = np.zeros(shape=(num_tracks, d_max, num_features), dtype=np.float32)
distance = np.zeros(shape=(num_tracks, d_max), dtype=np.float32)
metadata = []
i = 0
for filename in npz_files:
    # Progress
    sys.stdout.write("\rProgress ... {0:.2f} %".format(((i+1)/num_tracks)*100))
    sys.stdout.flush()
    
    matrix_temp, dst_temp, _ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, filename))

    # Imputate NaNs
    matrix_temp, _ = ml_utilities.imputate_nans_feature_matrix(matrix_temp, method='Interpolate', drop_nan=False)
    # Concatenate NaNs
    matrix_temp = ml_utilities.concat_nans_1d(matrix_temp, N=d_max)
    
    dst_temp = ml_utilities.concat_nans_1d(dst_temp, N=d_max)
    
#    pdb.set_trace()
#    matrix_temp = matrix_temp.fillna(value=0)
    label_temp = np.array(matrix_temp['SSHA_105'])
    matrix_temp = matrix_temp.drop(columns=['SSHA_35', 'SSHA_71', 'SSHA_105'])
#    matrix_temp = matrix_temp.drop(columns='ADG443_NN_OLCI_5km')
#    matrix_temp = matrix_temp.drop(columns='KD490_M07_OLCI_5km')
#    matrix_temp = matrix_temp.drop(columns='TSM_NN_OLCI_5km')
#    matrix_temp = matrix_temp.drop(columns='CHL_OC4ME_OLCI_5km')
#    matrix_temp = matrix_temp.drop(columns='ADG443_NN_OLCI_50km')
#    matrix_temp = matrix_temp.drop(columns='KD490_M07_OLCI_50km')
#    matrix_temp = matrix_temp.drop(columns='TSM_NN_OLCI_50km')
#    matrix_temp = matrix_temp.drop(columns='CHL_OC4ME_OLCI_50km')
#    pdb.set_trace()
    
    matrix_temp = np.array(matrix_temp, dtype=np.float32)
#    matrix_temp = np.convolve(matrix_temp.squeeze(), np.ones((95))/95, mode='same')
    label_temp = np.expand_dims(label_temp, axis=1)
#    matri_temp = np.expand_dims(matrix_temp, axis=1)
    matrix_temp = ml_utilities.my_standardizer(matrix_temp, matrix_temp) # standardize
    label_temp = ml_utilities.my_standardizer(label_temp, label_temp) # standardize
    
#    matrix_temp = matrix_temp.squeeze()
    label_temp = label_temp.squeeze()
    
#    matrix_temp = ml_utilities.matrix_min_max_rescale(matrix_temp, ub=0.5, lb=-0.5, axis=0) # rescale
#    matrix_temp = ml_utilities.matrix_min_max_rescale(matrix_temp, ub=np.nanmax(label_temp), lb=np.nanmin(label_temp), axis=0) # rescale
#    pdb.set_trace()
    matrix_temp[np.isnan(matrix_temp)] = 0 # nans to zeros
#    matrix_temp[np.isnan(matrix_temp)] = -2 # nans to zeros
#    label_temp[np.isnan(label_temp)] = 0 # nans to zeros
#    pdb.set_trace()
    label[i, :] = label_temp
    matrix[i,:,:] = matrix_temp
    distance[i, :] = dst_temp
    
    i = i + 1
    
del matrix_temp, label_temp, i

label = np.expand_dims(label, axis=2)
# =============================================================================
# MY LOSS FUNCTIONS
# =============================================================================
def my_loss(y_true, y_predict):
    # mask out nans
    idx = tf.math.logical_not(tf.math.is_nan(y_true))
#    idx_y_predict = tf.math.logical_not(tf.math.is_nan(y_predict))
#    idx = tf.math.logical_and(idx_y_true, idx_y_predict)
    
    y_true = tf.boolean_mask(y_true, idx)
    y_predict = tf.boolean_mask(y_predict, idx)
    
    # Compute MSE
    mse = keras.backend.mean(keras.backend.square(y_true - y_predict))
    
    return keras.backend.sqrt(mse)

def charbonnier(y_true, y_predict):
    epsilon = 1e-3
    error = y_true - y_predict
    p = keras.backend.sqrt(keras.backend.square(error) + keras.backend.square(epsilon))
    
    return keras.backend.mean(p)

def my_mae(y_true, y_predict):
    # mask out nans
    idx = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.boolean_mask(y_true, idx)
    y_predict = tf.boolean_mask(y_predict, idx)
    
    # Compute MSE
    mae = keras.backend.mean(tf.math.abs(y_true - y_predict))
    
    return mae

def my_huber_loss(y_true, y_predict):
    # mask out nans
    idx = tf.math.logical_not(tf.math.is_nan(y_true))
#    idx_y_predict = tf.math.logical_not(tf.math.is_nan(y_predict))
#    idx = tf.math.logical_and(idx_y_true, idx_y_predict)
    
    y_true = tf.boolean_mask(y_true, idx)
    y_predict = tf.boolean_mask(y_predict, idx)
    return tf.losses.huber_loss(y_true, y_predict)
    

# =============================================================================
# SETUP NETWORK
# =============================================================================

# SPLIT train-test randomly
x_train, x_test, label_train, label_test, distance_train, distance_test, npz_fname_train, npz_fname_test = sk.train_test_split(matrix, label, distance, npz_files, test_size=0.05)

#%%
# create model
model = keras.Sequential()

# 1st pack of Conv/Pooling
model.add(keras.layers.Conv1D(filters=3, kernel_size=3, input_shape=(d_max, num_features), padding='same')) # shape of input data
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.Conv1D(filters=3, kernel_size=3, strides=1, padding='same')) # shape of input data
#model.add(keras.layers.BatchNormalization())
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=7, strides=1, padding='same'))
#model.add(keras.layers.SpatialDropout1D(0.5))

### 1st pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same')) # shape of input data
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, strides=1, padding='same')) # shape of input data
##model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=5, strides=1, padding='same'))
##model.add(keras.layers.SpatialDropout1D(0.5))
#
## 2nd pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same')) # shape of input data
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same')) # shape of input data
##model.add(keras.layers.BatchNormalization())
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
##model.add(keras.layers.SpatialDropout1D(0.5))
#
### 3rd pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same')) # shape of input data
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same')) # shape of input data
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#
## 4th pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same'))
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#
## 4th pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same'))
#model.add(keras.layers.Conv1D(filters=3, kernel_size=5, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#
## 4th pack of Conv/Pooling
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#
#model.add(keras.layers.Conv1D(filters=10, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=1)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=1)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#model.add(keras.layers.Conv1D(filters=10, kernel_size=3, padding='same'))
#model.add(keras.layers.Conv1D(filters=10, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))
#
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
#model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
#model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
#model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

model.add(keras.layers.Conv1D(filters=3, kernel_size=3, padding='same'))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function
model.add(keras.layers.AveragePooling1D(pool_size=3, strides=1, padding='same'))

#model.add(keras.layers.Dense(units=800, use_bias=True))
#model.add(keras.layers.LeakyReLU(alpha=0.8))
#
#model.add(keras.layers.Dense(units=500, use_bias=True))
#model.add(keras.layers.LeakyReLU(alpha=0.8))
#
#model.add(keras.layers.Dense(units=200, use_bias=True))
#model.add(keras.layers.LeakyReLU(alpha=0.8))
#
model.add(keras.layers.Dense(units=15, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8))

# Add 4th layer (Fully Connected)
model.add(keras.layers.Dense(units=1, use_bias=True))


# Hyperparameters
batch_sz = 1
n_epochs = 15
learn_rate = 0.0001
lr_decay = learn_rate/n_epochs
#lr = learn_rate/(1 + lr_decay*np.arange(n_epochs)*num_tracks/batch_sz)
#plt.plot(np.arange(n_epochs), lr), plt.xlabel('# of Epochs'), plt.ylabel('Learning Rate')
#my_optim = keras.optimizers.RMSprop(lr=learn_rate)
#my_optim = keras.optimizers.Adam(lr=learn_rate)#, decay=lr_decay)
#my_optim = keras.optimizers.SGD(lr=0.001)
# Compile model
model.compile(loss=my_mae,#my_loss,#, # Loss function
              optimizer='rmsprop')

# Create Callback variables
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = keras.callbacks.ModelCheckpoint(os.path.join(models_path, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)

## non-nan values
#mask_useful = ~(x_test.mask).any(axis=2)

# Train model
history = model.fit(x=x_train, y=label_train,
          batch_size=batch_sz,
          epochs=n_epochs,
          verbose=1, # progress bar
          validation_split=0.2,
          shuffle=True,
          callbacks=[es, mc]) # fraction of data as validation

print(history.history.keys())
#%%
# =============================================================================
# PREDICT TRACK
model = keras.models.load_model(os.path.join(models_path, 'best_model.h5'), custom_objects={'my_loss': my_loss})
# =============================================================================
RMSE_test = []

for i in range(len(label_test)):
    track_number = i
    
    # Progress
    sys.stdout.write("\rProgress ... {0:.2f} %".format(i/(len(label_test)-1)*100))
    sys.stdout.flush()
    
    # TEST SET PREDICTION
    y_hat_test = model.predict(np.expand_dims(x_test[track_number, :, :], axis=0), verbose=0)
    y_hat_test = np.squeeze(y_hat_test)
    original_test = np.squeeze(label_test[track_number, :, :])
    dist_test = np.squeeze(distance[track_number, :])
    
    # =============================================================================
    # PLOT
    # =============================================================================
    idx = ~np.isnan(original_test)
    RMSE_test.append(np.sqrt(mean_squared_error(original_test[idx], y_hat_test[idx])))
    
    font = {'size' : 18}
    plt.rc('font', **font)
    fig, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
    ax1.scatter(dist_test[idx], original_test[idx], s=3)
    ax1.scatter(dist_test[idx], y_hat_test[idx],c='red', s=3)
    ax1.set_ylabel('SSHA [m]', fontsize=18)
    ax1.set_xlabel('Distance [m]', fontsize=18)
    ax1.legend(['Test_label', 'Test_predicted'], loc='upper left', fontsize=16)
    ax1.set_title(npz_fname_test[i][:-4], fontsize=23)
    
    # REGRESSION TEST
#    idx_nanan=np.isnan(original_test) | np.isnan(y_hat_test)
    res_thsen= scipy.stats.theilslopes(original_test[idx], y_hat_test[idx], 0.95)
    res_ols = scipy.stats.linregress(original_test[idx], y_hat_test[idx])
    
    ax2.scatter(original_test[idx], y_hat_test[idx])
    # Theil-Sen regression
    ax2.plot(original_test[idx], res_thsen[0]*original_test[idx] + res_thsen[1], 'r-')
    # Linear Least Squares regression
    ax2.plot(original_test[idx], res_ols[0]*original_test[idx] + res_ols[1], '#59ff00', linewidth=2.5)
    ax2.legend(['Theil-Sen', 'OLS'], loc='lower right', fontsize=16)
    ax2.axis('equal')
    ax2.set_xlabel('Test Label SSHA [m]')
    ax2.set_ylabel('Test Predicted SSHA [m]')
    #plt.xlim([-1, 1])
    ax2.text(0.2, -0.1, 'Theis-Sen\n'+'y = {0:.4f} x + {1:.4f}\nOLS\ny = {2:.4f} x + {3:.4f}'.format(res_thsen[0], res_thsen[1],res_ols[0],res_ols[1]), fontsize=13,
            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
    
    # Save plot     
    plotpath = r"C:\Users\vlachos\Desktop\SSTlevel4".replace('\\','\\')
    fig.savefig(os.path.join(plotpath, npz_fname_test[i][:-4]) + '.png', dpi=300)
    plt.close('all')

# TRAIN SET PREDICTION
RMSE_train = []
for track_number in range(len(label_train)):
    y_hat_train = model.predict(np.expand_dims(x_train[track_number, :, :], axis=0), verbose=0)
    y_hat_train = np.squeeze(y_hat_train)
    original_train = np.squeeze(label_train[track_number, :, :])
    
    idx = ~np.isnan(original_train)
    RMSE_train.append(np.sqrt(mean_squared_error(original_train[idx], y_hat_train[idx])))
    
fig2, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_ylabel('Loss [m]', fontsize=18)
ax1.set_xlabel('Epoch', fontsize=18)
ax1.set_xlim(0, len(history.history['loss']))
ax1.legend(['Train', 'Validation'], loc='upper left', fontsize=16)

RMSE_test = np.array(RMSE_test) # Convert to ndarray
RMSE_train = np.array(RMSE_train) # Convert to ndarray

ax2.hist(RMSE_train, bins=15, color='orange')
ax2.hist(RMSE_test, bins=15, fill=False, edgecolor='black')
ax2.set_xlabel('RMSE [m]', fontsize=18)
ax2.set_ylabel('# of counts', fontsize=18)
ax2.set_title('SSHA 1D CNN RMSEs', fontsize=23)
ax2.legend(['Train', 'Test'], fontsize=16)
#ax2.text(0.20,1000 , 'Features:\nSST versions', fontsize=13,
#         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

fig2.savefig(os.path.join(plotpath, 'Train_Validation_Test_Loss_RMSE') + '.png', dpi=300)

plt.close('all')

for i in range(len(label_test)):
    fig2 = plt.figure(figsize=(10,8))
    plt.plot(distance_test[i,:], label_test[i,:,0], distance[i,:], x_test[i,:, 1])
    plt.legend('SSHA original', 'SST')
    plt.title('TEST')
    fig2.savefig(os.path.join(plotpath, 'SSHA_SST {0}'.format(i)) + '.png', dpi=300)

    plt.close('all')
    