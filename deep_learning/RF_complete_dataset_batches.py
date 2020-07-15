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
import scipy.stats
from sklearn.metrics import mean_squared_error
import os
import pandas as pd
import pdb
import sys
import sklearn.model_selection as skmodel_selection
import sklearn.ensemble as skensemble
import time
# My Modules
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
# Import a lot of trajectories
filespath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr'.replace('\\','\\')

npz_files = os.listdir(filespath) # npz files names
npz_files = [item for item in npz_files if '.npz' in item]
#npz_files = ml_utilities.pick_npz_dates(npz_files, '2018-05-10', n_plus=10) # Select certain timespan only
npz_files_batches = ml_utilities.split_npz_files_by_dates(npz_files, n_plus=5) # Split dataset in 10-day batches
N_npz_files_batches = len(npz_files_batches)

counter_1 = 1
counter_3 = 1
# Derive names of variables
variable_names = np.load(os.path.join(filespath, npz_files[0]), encoding='latin1', allow_pickle=True)
# Retrieve dictionary
variable_names = variable_names['arr_0'].item()
del variable_names['Metadata'], variable_names['SSHA_35'], variable_names['Distance']
del npz_files
# Initialize RF Model
# Define hyperparameters
params = {'n_estimators': 150, 
          'criterion': 'mse', # Or 'mae'
          'max_depth': 10, # of the tree
          'min_samples_leaf': 7,
          'bootstrap':True,
          'max_features': 'sqrt',
          'verbose': 0
          }
model = skensemble.RandomForestRegressor(**params)

# Initialize bad files list
bad = []

# Initializes RMSEs
RMSE_train = []
RMSE_test = []
RMSE_unseen_track = []
RMSE_ols = []
for npz_files in npz_files_batches:
    # Progress
    print('Batch: {0} out of {1}\n'.format(counter_3, N_npz_files_batches))
    N_npz_files = len(npz_files)
    counter_1 = 1
    for unseen_track in  npz_files:
        
        # Progress
        print('    Files: {0} out of {1}\n'.format(counter_1, N_npz_files))
            
        temp_npz_files = npz_files.copy() # Keep list of files in a temporary variable
        pdb.set_trace()
        temp_npz_files.remove(unseen_track) # Remove unseen track from temporary dataset
        
        # Initialize matrix and label
        matrix = pd.DataFrame(columns=list(variable_names.keys()), dtype=np.float32)
        label = pd.DataFrame(columns=['SSHA_35'], dtype=np.float32)
        
        counter_2 = 0
        N_temp_npz_files = float(len(temp_npz_files)) # Number of temp npz files where the training is going to be based on
        try:
            for file_name in temp_npz_files:
        
                # Progress
                sys.stdout.write("\rProgress ... {0:.2f} %".format((counter_2/N_temp_npz_files)*100))
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
                
                if True: # 1) IMPUTATION OF NANs- INTERPOLATION AND DROP THE REST OF THE NANS
                    
                    # Interpolate the NAN values inside the dataset
                    data_temp = data_temp.interpolate(method='akima', limit=150, limit_direction='both', axis=0)
                    # Interpolate (actually extrapolate) the values at the edges
                    data_temp = data_temp.interpolate(method='linear', limit=50, limit_direction='both', axis=0)
                    # Delete remaining rows with NaNs
                    data_temp = data_temp.dropna()
                    
                elif False: # 2) IMPUTATION OF NANs- ZEROS
                    
                    # Replace NaN with 0 in the label dataset
                    data_temp = data_temp.fillna(value=0)
                
                elif False: # 3) DELETE ROWS WITH NANs
            
                    # Delete row with NaN in the label dataset
                    data_temp = data_temp.dropna()
               
                # Concatenate label
                label = pd.concat([label, pd.DataFrame(data_temp['SSHA_35'])])
                # Delete SSHA column and keep the SST columns
                data_temp = data_temp.drop(columns=['SSHA_35'])
                
                # Concatenate features (SST) to matrix
                matrix = pd.concat([matrix, data_temp])
            
                counter_2 = counter_2 + 1
        
            del data_temp
            
#            ub = 1
#            lb = -1
#            varmin = matrix.min(axis=0)
#            varmax = matrix.max(axis=0)
#            matrix = lb + (matrix - varmin)/(varmax - varmin) * (ub - lb)
        
    
            
            # Converta pandas to arrays
            matrix = np.array(matrix)
            label = np.array(label).squeeze()
            
            # SPLIT train-test randomly
            x_train, x_test, label_train, label_test = skmodel_selection.train_test_split(matrix, label, test_size=0.30)
            
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
            time_start = time.time()
                   
            # Fit model to train data
            x_train = pd.DataFrame(x_train) 
            x_test = pd.DataFrame(x_test) 
            model.fit(x_train, label_train)
            
            time_end = time.time()
            
            print('\nTraining and fitting took {0} seconds\n'.format(time_end - time_start))
            
            # Predict on test data
            y_hat_test = model.predict(x_test)
            y_hat_train = model.predict(x_train)
            
            # Compute RMSE
            RMSE_train.append(np.sqrt(mean_squared_error(y_hat_train, label_train)))
            RMSE_test.append(np.sqrt(mean_squared_error(y_hat_test, label_test)))
            
            # =============================================================================
            # APPLY MODEL TO UNSEEN DATA (NEW TRACK WHICH IS NOT INCLUDED IN TRAINING)
            # =============================================================================
            # Predict on unknown data (trajectory that kept out of the npz_files in the beginning)
            # Initialize matrix and label
            dat = np.load(os.path.join(filespath, unseen_track), encoding='latin1', allow_pickle=True)
            # Retrieve dictionary
            dat = dat['arr_0'].item()
            
            del dat['Metadata']
            
            data = pd.DataFrame.from_dict(dat, dtype=np.float32)
            
            # Interpolate the NAN values inside the dataset
            data = data.interpolate(method='akima', limit=150, limit_direction='both', axis=0)
            # Interpolate (actually extrapolate) the values at the edges
            data = data.interpolate(method='linear', limit=50, limit_direction='both', axis=0)
            # Delete remaining rows with NaNs
            data = data.dropna()
            
            if data.size == 0:
                N_npz_files = N_npz_files - 1
                bad.append(unseen_track)
                continue
            
            original = data['SSHA_35']
            distance = data['Distance']
            data = data.drop(columns=['Distance', 'SSHA_35']) # Kick-out Distance and SSHA_35
#            data = data.drop(columns=['SSHA_35']) # Kick-out SSHA_35 and KEEP Distance
            
    #        # Apply Standardizer and PCA
    #        data_temp = data.copy()
    #        data = ml_utilities.my_standardizer(data, data_temp)
    #        data_temp = data.copy()
    #        data = ml_utilities.my_pca(data, data, prin_comp=5)
    #        del data_temp
            
            # Normalize
#            varmin = data.min(axis=0)
#            varmax = data.max(axis=0)
#            data = lb + (data - varmin)/(varmax - varmin) * (ub - lb)
            
            # Use model to predict test data
            y_hat = model.predict(data)
            
            # Compute RMSE of test (unseen)
            RMSE_unseen_track.append(np.sqrt(mean_squared_error(y_hat, original)))
            
            # =============================================================================
            # REGRESSION ON UNSEEN TRACK PREDICTED/ORIGINAL     
            # =============================================================================
            res_ols = scipy.stats.linregress(label_test.squeeze(), y_hat_test)
            rmse_ols = np.sqrt(mean_squared_error(label_test.squeeze(), res_ols[0]*y_hat_test + res_ols[1]))
            RMSE_ols.append(rmse_ols)
        
            # =============================================================================
            # PLOT    
            # =============================================================================
            font = {'size' : 18}
            plt.rc('font', **font)
            fig, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
            
        #    ax1.figure(figsize=(12,8))
            ax1.scatter(label_test, y_hat_test)
            ax1.set_xlabel('Label [m]', fontsize=18)
            ax1.set_ylabel('Predicted [m]', fontsize=18)
            ax1.set_title('Test data', fontsize=23)
            ax1.text(-0.4, 0.4, 'RMSE = {0:.4f}\nOLS\ny = {1:.4f} x + {2:.4f}'.format(rmse_ols, res_ols[0], res_ols[1]), fontsize=13,
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
            
            plt.figure(figsize=(12,8))
            plt.scatter(distance, original, s=3)
            plt.scatter(distance, y_hat, s=3)
            plt.legend(['Original', 'Predicted'], loc='best')
            plt.xlabel('Distance [m]', fontsize=18)
            plt.ylabel('SSHA [m]', fontsize=18)
            plt.title('Unseen Track {0}'.format(unseen_track[:-4]), fontsize=23)
            
        #    ax2.figure(figsize=(12,8))
            ax2.scatter(original, y_hat, s=3)
            ax2.set_xlabel('Original [m]', fontsize=18)
            ax2.set_ylabel('Predicted [m]', fontsize=18)
            ax2.set_title('Unseen Track {0}'.format(unseen_track[:-4]), fontsize=23)
        
            # Save plot
            
            plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_sral_slstr_batches'.replace('\\','\\')
            plt.savefig(os.path.join(plotpath, unseen_track[:-4]) + '.png', dpi=300)
            fig.savefig(os.path.join(plotpath, unseen_track[:-4]) + '_traj' + '.png', dpi=300)
            
            plt.close('all')
            
            counter_1 = counter_1 + 1
        #    pdb.set_trace()
            # Clear variables
            del label_test, label_train, matrix, x_train, x_test, y_hat, y_hat_test, y_hat_train, data, varmin, varmax
        except:
            N_npz_files = N_npz_files - 1
            bad.append(unseen_track)
            continue
    counter_3 = counter_3 + 1
    
# Convert RMSES to ndarray
RMSE_train = np.array(RMSE_train)
RMSE_test = np.array(RMSE_test)
RMSE_unseen_track = np.array(RMSE_unseen_track)
RMSE_ols = np.array(RMSE_ols)
# Save RMSEs and bad files
out_errors = {'RMSE_train': RMSE_train, 'RMSE_test': RMSE_test,
              'RMSE_unseen_track': RMSE_unseen_track, 'RMSE_ols': RMSE_ols, 'bad_files': bad
              }
save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_sral_slstr_batches'.replace('\\','\\')
np.savez_compressed(os.path.join(save_path, 'RF_leave1out_sral_slstr_10dayBatches.npz'), out_errors)

# Plot RMSEs histogram      
font = {'size' : 18}
plt.rc('font', **font)     
plt.figure(figsize=(10, 8))
plt.hist(RMSE_unseen_track, bins=int(round(RMSE_unseen_track.size**(1/3.0)*2)), color='b', alpha=0.6)
plt.hist(RMSE_train, bins=4, color='orange')
plt.hist(RMSE_test, bins=4, fill=False, edgecolor='black')
plt.xlabel('Leave-one-out RMSE [m]', fontsize=18)
plt.ylabel('# of counts', fontsize=18)
plt.title('SSHA Random Forest predictions 10-day batches', fontsize=23)
plt.legend(['New Track', 'Train', 'Test'], fontsize=13)
plt.text(0.20,1000 , 'Features:\nSST versions', fontsize=13,
         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})
