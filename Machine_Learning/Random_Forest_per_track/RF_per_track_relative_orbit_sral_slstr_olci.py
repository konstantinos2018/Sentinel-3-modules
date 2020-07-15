# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as skmodel_selection
from sklearn.metrics import mean_squared_error
import sklearn.ensemble as skensemble
import scipy.stats
import pandas as pd
import os
import sys
import pdb
import pickle
import datetime as dt
# My Modules
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
#plt.ioff()
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr_olci_RF'.replace('\\','\\')
npz_files = os.listdir(path)
N_npz_files = len(npz_files)

# Define hyperparameters
params = {'n_estimators': 6, 
          'criterion': 'mse', # Or 'mae'
          'max_depth': 9, # of the tree
          'min_samples_leaf': 3,
          'bootstrap':True,
          'max_features': 'sqrt',
          'verbose': 0
          }

model = skensemble.RandomForestRegressor(**params)
i = 1
bad = []
RMSE_test = []
RMSE_train = []
importance_list = []
for filename in npz_files:    
    try:
        plt.close('all')
        # Progress
        sys.stdout.write("\rFiles {0} out of {1}".format(i, N_npz_files))
        sys.stdout.flush()
        
        fullpath = os.path.join(path, filename)
        matrix, distance, metadata = ml_utilities.feature_matrix_from_npz(fullpath)
    
        # =============================================================================
        # MISSING VALUES IMPUTATION            
        # =============================================================================
        matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=True)
        
        label = np.array(matrix['SSHA_35'])
        
#        matrix = matrix.drop(columns='SSHA_35')
#        matrix_labels_out = [item for item in matrix.columns.to_list() if ('OLCI' not in item) or ('CHL' not in item)]
#        matrix_labels_out = ['SSHA_35', 'SST_12.5km', 'SST_32km', 'SST_125km']
        matrix_labels_out = ['SSHA_35', 'KD490_M07_OLCI_12.5km', 'CHL_OC4ME_OLCI_5km', 'KD490_M07_OLCI_5km',
                             'ADG443_NN_OLCI_5km', 'TSM_NN_OLCI_12.5km', 'CHL_OC4ME_OLCI_12.5km',
                             'TSM_NN_OLCI_5km', 'CHL_OC4ME_OLCI_12.5km', 'SST_5km', 'TSM_NN_OLCI_12.5km',
                             'ADG443_NN_OLCI_12.5km']
#        matrix_labels_out = ['SSHA_35']
        matrix = matrix.drop(columns=matrix_labels_out)
        matrix_labels = matrix.columns.to_list() # keep feature matrix names
        break
        matrix = np.array(matrix)
        
        # =============================================================================
        # SETUP RANDOM FOREST
        # =============================================================================
        
        # Create array of indices
        idx = np.arange(label.size)
        
        # SPLIT train-test randomly
        x_train, x_test, label_train, label_test, idx_train, idx_test = skmodel_selection.train_test_split(matrix, label, idx, test_size=0.30)
        
        # Fit model to train data
        model.fit(x_train, label_train)
        
        # Use model to predict test data
        y_hat_test = model.predict(x_test)
        # Train set validation
        y_hat_train = model.predict(x_train)
        
        # Compute RMSE of test (unseen)
        RMSE_test.append(np.sqrt(mean_squared_error(label_test, y_hat_test)))
        RMSE_train.append(np.sqrt(mean_squared_error(label_train, y_hat_train)))
        
        importances = model.feature_importances_
        importance_list.append(np.expand_dims(importances, axis=1))
        # =============================================================================
        # PLOT
        # =============================================================================
        font = {'size' : 18}
        plt.rc('font', **font)
        fig, [ax1, ax2] = plt.subplots(2,1, sharex=False, figsize=(13,16))
        ax1.scatter(distance[idx_test], label_test, s=3)
        ax1.scatter(distance[idx_test], y_hat_test, c='red', s=3)
        ax1.set_xlabel('Distance [m]', fontsize=18)
        ax1.set_ylabel('SSHA [m]', fontsize=18)
        ax1.set_title(filename[:-4], fontsize=23)
        ax1.legend(['Test_label', 'Test_predicted'], loc='upper left', fontsize=16)
    
        # OLS REGRESSION
        res_ols = scipy.stats.linregress(label_test, y_hat_test)
    
        ax2.scatter(label_test, y_hat_test)
        ax2.plot(label_test, res_ols[0]*label_test + res_ols[1], '#59ff00', linewidth=2.5)
        ax2.set_xlabel('Test Label SSHA [m]', fontsize=18)
        ax2.set_ylabel('Test Predicted SSHA [m]', fontsize=18)
        ax2.text(0.1, 0.8, 'OLS\ny = {0:.4f} x + {1:.4f}'.format(res_ols[0],res_ols[1]), fontsize=13,
                 bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, transform=ax2.transAxes)
    
        fig2 = plt.figure(figsize=(12,8))
    #    font = {'family' : 'normal','weight' : 'normal','size' : 14}
    #    plt.rc('font', **font)
#        ax = fig2.gca()
        plt.title(filename[:-4], fontsize=23)
        plt.ylabel('%', fontsize=18)
        plt.bar(range(len(matrix_labels)), importances*100, color='b', align="center")
        plt.xticks(range(len(matrix_labels)), matrix_labels, rotation = 55, fontsize=5)
        
        plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_olci_slstr_PerTrack'.replace('\\','\\')
        fig.savefig(os.path.join(plotpath, filename[:-4]) + '.png', dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(plotpath, filename[:-4]) + '_FeatImportance' + '.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        plt.close(fig2)
        
        # Save model
        model_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_olci_slstr_PerTrack\models'.replace('\\','\\')
        olci_date = metadata.split(' ')[3]
        olci_date = dt.datetime.strptime(olci_date, '%Y%m%dT%H%M%S').strftime('%Y-%m-%d %H_%M_%S')
        model_name = '{0}__{1}_RF_slstr_olci_model.sav'.format(filename[:-4], olci_date)
        pickle.dump(model, open(os.path.join(model_path, model_name), 'wb'))
    
        i = i + 1
        
#        del fig, fig2, ax1, ax2, ax, x_train, x_test, label_train, label_test
    except:
        plt.close('all')
        N_npz_files = N_npz_files - 1
        bad.append(filename)
        continue

importance_list = np.array(importance_list)
importance_list = importance_list.squeeze()
importance_list = np.mean(importance_list, axis=0) # mean of every importance of feature

RMSE_test = np.array(RMSE_test)
RMSE_train = np.array(RMSE_train)

fig3 = plt.figure(figsize=(10, 8))
plt.hist(RMSE_train, bins=15, color='orange')
plt.hist(RMSE_test, bins=15, fill=False, edgecolor='black')
plt.xlabel('RMSE [m]', fontsize=18)
plt.ylabel('# of counts', fontsize=18)
plt.title('SSHA Random Forest', fontsize=23)
plt.legend(['Train', 'Test'], fontsize=13)
plt.text(0.015, 40, 'Features:\n'+'\n'.join(matrix_labels), fontsize=13,
         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

fig3 = plt.figure(figsize=(12,8))
plt.title(filename[:-4], fontsize=23)
plt.ylabel('%', fontsize=18)
plt.bar(range(len(matrix_labels)), importance_list*100, color='b', align="center")
plt.xticks(range(len(matrix_labels)), matrix_labels, rotation = 55, fontsize=5)
fig3.savefig(os.path.join(plotpath, filename[:-4]) + '_Mean_FeatImportance' + '.png', dpi=300, bbox_inches='tight')       
