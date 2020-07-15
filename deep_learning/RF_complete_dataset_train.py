# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


# =============================================================================
# MULTIPLE TRAJECTORIES INPUT
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
#import tensorflow as tf
#import tensorflow.keras as keras
import scipy.stats
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import pdb
import sys
import sklearn.model_selection as skmodel_selection
import sklearn.ensemble as skensemble
import time
import pickle
# My Modules
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
# Import a lot of trajectories

filespath = r"D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\grid_npz_sral_slstr_olci_RF\RF_complete_dataset_model_sral_slstr_olci".replace('\\','\\')

npz_files = os.listdir(filespath)
npz_files = [item for item in npz_files if item[-4:] == '.npz']
N_npz_files = len(npz_files)

counter_1 = 0

# Derive names of variables
variable_names = np.load(os.path.join(filespath, npz_files[0]), encoding='latin1', allow_pickle=True)
# Retrieve dictionary
variable_names = variable_names['arr_0'].item()
del variable_names['Metadata'], variable_names['SSHA_35'], variable_names['Distance']

# Initialize RF Model
# Define hyperparameters

params = {'n_estimators': 200, 
          'criterion': 'mse', # Or 'mae'
          'max_depth': 10, # of the tree
          'min_samples_leaf': 7,
          'bootstrap':True,
          'max_features': 'sqrt',
          'verbose': 0
          }
model = skensemble.RandomForestRegressor(**params)

time_start = time.time()
# Progress
print('Files: {0} out of {1}\n'.format(counter_1, N_npz_files))
    
temp_npz_files = npz_files.copy() # Keep list of files in a temporary variable
#    temp_npz_files.remove(unseen_track) # Remove unseen track from temporary dataset

# Initialize matrix and label
matrix = pd.DataFrame(columns=list(variable_names.keys()), dtype=np.float32)
label = pd.DataFrame(columns=['SSHA_35'], dtype=np.float32)

counter_2 = 0

for file_name in temp_npz_files:

    # Progress
    sys.stdout.write("\rProgress ... {0:.2f} %".format((counter_2/N_npz_files)*100))
    sys.stdout.flush()
            
    # load npz files. encoding argument is used only if npz files have been
    # saved using py2.x and are loaded by py3.x
    dat = np.load(os.path.join(filespath, file_name), encoding='latin1', allow_pickle=True)
    
    # Retrieve dictionary
    dat = dat['arr_0'].item()
    # Keep distance in variable
#    distance = dat['Distance']
    del dat['Metadata'], dat['Distance']
    
    # =============================================================================
    # MISSING VALUES HANDLING            
    # =============================================================================
    # Assign label and feature matrix to temporary variables
    data_temp = pd.DataFrame.from_dict(dat, dtype=np.float32)
    
    data_temp = ml_utilities.imputate_nans_feature_matrix(data_temp, method='Interpolate', drop_nan=True)
   
    # Concatenate label
    label = pd.concat([label, pd.DataFrame(data_temp['SSHA_35'])])
    # Delete SSHA column and keep the SST columns
    data_temp = data_temp.drop(columns=['SSHA_35'])
    
    # Concatenate features (SST) to matrix
    matrix = pd.concat([matrix, data_temp], axis=0)

    counter_2 = counter_2 + 1

del data_temp

# Rescale
ub = 1
lb = -1
matrix = ml_utilities.matrix_min_max_rescale(matrix, ub=ub, lb=lb, axis=0)

# Converta pandas to arrays
matrix = np.array(matrix)
label = np.array(label).squeeze()


#        # =============================================================================
#        # RESCALE AND APPLY PCA
#        # =============================================================================
#        x_train_temp = x_train.copy()
#        x_train = ml_utilities.my_standardizer(x_train, x_train_temp)
#        x_test = ml_utilities.my_standardizer(x_test, x_train_temp)
#        
#        x_train = pd.DataFrame(x_train)
#        x_test = pd.DataFrame(x_test)
#        
#        x_train_temp = x_train.copy()
#        x_train = ml_utilities.my_pca(x_train, x_train_temp, prin_comp=5)
#        x_test = ml_utilities.my_pca(x_test, x_train_temp, prin_comp=5)
#        del x_train_temp

# =============================================================================
# SETUP RANDOM FOREST MODEL
# =============================================================================

model.fit(matrix, label)

# Clear variables
del matrix, label
time_end = time.time()
print('\nFeature matrix built, Training and Fitting took {0} seconds\n'.format(time_end - time_start))

save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_complete_sral_slstr_olci'.replace('\\','\\')
filename = 'RF_complete_data_model.sav'
pickle.dump(model, open(os.path.join(save_path, filename), 'wb'))

