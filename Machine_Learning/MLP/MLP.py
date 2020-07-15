# -*- coding: utf-8 -*-

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
import sys
import pandas as pd
import pdb
# My Modules
from S3postproc import rescale_between
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
filespath = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr'.replace('\\','\\')
npz_files = os.listdir(filespath)
npz_files = [filename for filename in npz_files if '.npz' in filename]
N_npz_files = len(npz_files)
models_path = r'C:\Users\vlachos\Desktop\MLP'.replace('\\','\\')

# Derive names of variables
var_to_drop = ['SSHA_35', 'SST_125km', 'SST_95km','SST_75km', 'SST_32km', 'SST_16km', 'SST_12.5km']
matrix, _, _ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, npz_files[0]))
matrix = matrix.drop(columns=var_to_drop)
matrix_labels = list(matrix.columns) # keep feature matrix names
del matrix

i = 1
bad = []
RMSE_test = []
RMSE_train = []
matrix = pd.DataFrame(columns=matrix_labels, dtype=np.float32)
label = pd.DataFrame(dtype=np.float32)

for filename in npz_files:    
    try:
        
        # Progress
        sys.stdout.write('\rFiles {0} out of {1}'.format(i, N_npz_files))
        sys.stdout.flush()
        
        fullpath = os.path.join(filespath, filename)
        matrix_temp, distance, _ = ml_utilities.feature_matrix_from_npz(fullpath)
        
        # =============================================================================
        # MISSING VALUES IMPUTATION            
        # =============================================================================
        matrix_temp, _ = ml_utilities.imputate_nans_feature_matrix(matrix_temp, method='Interpolate', drop_nan=True)
        
        label_temp = matrix_temp['SSHA_35']
        
        matrix_temp = matrix_temp.drop(columns=var_to_drop)
        
        # Concatenate features (SST) to matrix
        label = pd.concat([label, label_temp], axis=0)

        matrix = pd.concat([matrix, matrix_temp], axis=0, ignore_index=True)        
#        if i == 5:
#            break
        i = i + 1
    except:
        print('STOPPED')
        N_npz_files = N_npz_files - 1
        continue

del label_temp, matrix_temp
# Convert to ndarrays
label = np.array(label)
matrix = np.array(matrix)
#%%
# =============================================================================
# MY LOSS FUNCTIONS
# =============================================================================
def my_loss(y_true, y_predict):
    # Compute squared error
    error = (y_predict - y_true)**2

    # Compute MSE
    mse = keras.backend.mean(error)
    
    return keras.backend.sqrt(mse)

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

# SPLIT train-test randomly
x_train, x_test, label_train, label_test = sk.train_test_split(matrix, label, test_size=0.30)

# create model
model = keras.Sequential()

# Packs of Fully-connected layers
model.add(keras.layers.Dense(units=256, use_bias=True, input_shape=(4,)))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=9*16, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=8*16, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Add de-noise layer
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units=64, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=32, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=16, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=8, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=16, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=32, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=64, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=128, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=256, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

model.add(keras.layers.Dense(units=1, use_bias=True))
model.add(keras.layers.LeakyReLU(alpha=0.8)) # activation function

# Compile model
model.compile(loss=tf.losses.huber_loss,#my_loss,#'mse', # Loss function
              optimizer='adam')
              # metrics=['mean_squared_error']) # Optimizer

# Create Callback variables
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = keras.callbacks.ModelCheckpoint(os.path.join(models_path, 'best_model.h5'), monitor='val_loss', mode='min', save_best_only=True, verbose=1)

## non-nan values
#mask_useful = ~(x_test.mask).any(axis=2)

# Train model
history = model.fit(x=x_train, y=label_train,
          batch_size=5,
          epochs=100,
          verbose=1, # progress bar
          validation_split=0.2,
          shuffle=True,
          callbacks=[es, mc]) # fraction of data as validation

print(history.history.keys())

#%%
saved_model = keras.models.load_model(os.path.join(models_path, 'best_model.h5'), custom_objects={'huber_loss': tf.losses.huber_loss})

for filename in npz_files:
    plt.close('all')
    
    # Progress
    sys.stdout.write("\rProgress ... {0:.2f} %".format(i/N_npz_files*100))
    sys.stdout.flush()
    
    fullpath = os.path.join(filespath, filename)
    data, distance, _ = ml_utilities.feature_matrix_from_npz(fullpath)
    
    data, idx_nan = ml_utilities.imputate_nans_feature_matrix(data, method='Interpolate', drop_nan=True)
    
    original = data['SSHA_35']
    
    data = data.drop(columns=var_to_drop)
    
    # convert to ndarray
    original = np.array(original)
    data = np.array(data)
    
    # TEST SET PREDICTION
    y_hat = model.predict(data, verbose=0)
#    dist_test = np.squeeze(distance[track_number, :])
    
    # =============================================================================
    # PLOT
    # =============================================================================
#    idx = ~np.isnan(original)
#    RMSE_test.append(np.sqrt(mean_squared_error(original[~idx_nan], y_hat)))
    
    font = {'size' : 18}
    plt.rc('font', **font)
    fig, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
    ax1.scatter(distance[~idx_nan], original, s=3)
    ax1.scatter(distance[~idx_nan], y_hat,c='red', s=3)
    ax1.set_ylabel('SSHA [m]', fontsize=18)
    ax1.set_xlabel('Distance [m]', fontsize=18)
    ax1.legend(['Label_unseen', 'Predicted_unseen'], loc='upper left', fontsize=16)
    ax1.set_title(filename[:-4], fontsize=23)
    
#    # REGRESSION TEST
##    idx_nanan=np.isnan(original_test) | np.isnan(y_hat_test)
#    res_thsen= scipy.stats.theilslopes(original_test[idx], y_hat_test[idx], 0.95)
#    res_ols = scipy.stats.linregress(original_test[idx], y_hat_test[idx])
#    
#    ax2.scatter(original_test[idx], y_hat_test[idx])
#    # Theil-Sen regression
#    ax2.plot(original_test[idx], res_thsen[0]*original_test[idx] + res_thsen[1], 'r-')
#    # Linear Least Squares regression
#    ax2.plot(original_test[idx], res_ols[0]*original_test[idx] + res_ols[1], '#59ff00', linewidth=2.5)
#    ax2.legend(['Theil-Sen', 'OLS'], loc='lower right', fontsize=16)
#    ax2.axis('equal')
#    ax2.set_xlabel('Test Label SSHA [m]')
#    ax2.set_ylabel('Test Predicted SSHA [m]')
#    #plt.xlim([-1, 1])
#    ax2.text(0.2, -0.1, 'Theis-Sen\n'+'y = {0:.4f} x + {1:.4f}\nOLS\ny = {2:.4f} x + {3:.4f}'.format(res_thsen[0], res_thsen[1],res_ols[0],res_ols[1]), fontsize=13,
#            bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
#    
    # Save plot     
    plotpath = r"C:\Users\vlachos\Desktop\MLP".replace('\\','\\')
    fig.savefig(os.path.join(plotpath, filename[:-4]) + '.png', dpi=300)
    plt.close('all')
    i = i + 1
    
## Predict on "unseen" data
#y_hat_test = model.predict(x_test)
#y_hat_test = np.squeeze(y_hat_test)
#
#y_hat_train = model.predict(x_train)
#y_hat_train = y_hat_train.squeeze()
## Predict on unseen data
#y_hat = model.predict(x_test)
#y_hat = np.squeeze(y_hat)

#idx_test = np.sort(idx_test)
# Plot prediction and label
font = {'size' : 18}
plt.rc('font', **font)
fig2, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
#ax1.scatter(distance[idx_test], label[:,0,0][idx_test])
#ax1.scatter(distance[idx_test], y_hat[idx_test],c='red', s=3)
#ax1.set_ylabel('SSHA [m]', fontsize=18)
#ax1.set_xlabel('Distance [m]', fontsize=18)
#ax1.legend(['Test_label', 'Test_predicted'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('Loss [m]', fontsize=18)
ax2.set_xlabel('Epoch', fontsize=18)
ax2.set_xlim(0, len(history.history['loss']))
ax2.legend(['Train', 'Validation'], loc='upper left')

# =============================================================================
# REGRESSIONS ON LABEL (TEST) AND PREDICTED (TEST)
# =============================================================================
#res_thsen= scipy.stats.theilslopes(label_test, y_hat_test, 0.95)
#res_ols = scipy.stats.linregress(label_test, y_hat_test)
#RMSE_test = np.sqrt(mean_squared_error(label_test, y_hat_test))
#
#plt.figure(figsize=(10,7))
#plt.scatter(label_test, y_hat_test)
## Theil-Sen regression
#plt.plot(label_test.squeeze(), res_thsen[0]*label_test.squeeze() + res_thsen[1], 'r-')
## Linear Least Squares regression
#plt.plot(label_test.squeeze(), res_ols[0]*label_test.squeeze() + res_ols[1], '#59ff00', linewidth=2.5)
#plt.legend(['Theil-Sen', 'OLS'], loc='lower right')
#plt.axis('equal')
#plt.xlabel('Label SSHA [m]')
#plt.ylabel('Predicted SSHA [m]')
##plt.xlim([-1, 1])
#plt.text(0.2, -0.1, 'Theis-Sen\n'+'y = {0:.4f} x + {1:.4f}\nRMSE = {2:.4f}\nOLS\ny = {3:.4f} x + {4:.4f}'.format(res_thsen[0], res_thsen[1], RMSE,res_ols[0],res_ols[1]), fontsize=13,
#        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
#
#train_loss = saved_model.evaluate(x_train, label_train, verbose=0)
#test_loss = saved_model.evaluate(x_test, label_test, verbose=0)
#print('Train loss: {0:.4f} [m]\nTest loss: {1:.4f} [m]'.format(train_loss, test_loss))

#json_string = model.to_json()
#with open(r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\1DCNN\cnn_13.json'.replace('\\','\\'), 'w') as json_file:  
#    json_file.write(json_string)
fig2.savefig(os.path.join(plotpath, filename[:-4]) + 'Loss_test_train' + '.png', dpi=300)
