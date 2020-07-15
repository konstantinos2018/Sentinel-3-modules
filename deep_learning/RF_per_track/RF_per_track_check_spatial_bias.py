# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import pdb
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
# My Modules
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
path_models = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_slstr_PerTrack\JustRight_SSTs\models'.replace('\\','\\')
models_files = os.listdir(path_models)
# Read npz list
path_npzfiles = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr'.replace('\\','\\')
npz_files = os.listdir(path_npzfiles)
npz_files = [item for item in npz_files if 'npz' in item]

model_name = 'S3B_2019-03-28 14_55_41__2019-03-28 01_16_43_RF_slstr_model.sav'

for npz in npz_files:
    if npz[4:14] == model_name[4:14]:
        pass
    else:
        continue
    
    # Read model
    model = pickle.load(open(os.path.join(path_models, model_name), 'rb'))
    # Read npz file
    matrix, distance, _ = ml_utilities.feature_matrix_from_npz(os.path.join(path_npzfiles, npz))
    matrix, idx_nan = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=True)
    
    label = np.array(matrix['SSHA_35'])

    matrix = matrix.drop(columns=['SSHA_35', 'SST_125km', 'SST_95km','SST_75km', 'SST_32km', 'SST_16km', 'SST_12.5km'])
    
    matrix_labels = list(matrix.columns) # keep feature matrix names
    matrix = np.array(matrix)
    
    # Predict
    y_hat = model.predict(matrix)
    
    # PLOT
    font = {'size' : 18}
    plt.rc('font', **font)
    fig = plt.figure(figsize=(13,16))
    plt.scatter(distance[~idx_nan], label, s=3)
    plt.scatter(distance[~idx_nan], y_hat,c='red', s=3)
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('SSHA [m]', fontsize=18)
    plt.title(npz[:-4], fontsize=23)
    plt.legend(['Test_label', 'Test_predicted'], loc='upper left', fontsize=16)
    
    plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_slstr_PerTrack\JustRight_SSTs\Bias_check\check_8'.replace('\\','\\')
    fig.savefig(os.path.join(plotpath, npz[:-4]) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
        