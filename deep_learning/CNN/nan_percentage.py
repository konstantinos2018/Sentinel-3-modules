# -*- coding: utf-8 -*-
# =============================================================================
# SCRIPT DESCRIPTION
# =============================================================================
# The script is used for the 1D CNN approach. It computes the percentage of
# missing values after concatenating the along-tracks with NaNs. It can also
# be used with minor change to compute the missing values without the concate-
# nation. It also computes the percentage of tracks that contain over 50%
# of missing values and plot it.

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
models_path = r"C:\Users\vlachos\Desktop\CNN_1D_real".replace('\\','\\')
filespath = r'C:\Users\vlachos\Desktop\npz_files_1DCNN_real'.replace('\\','\\')
npz_files = os.listdir(filespath)

# load npz files. encoding argument is used only if npz files have been
# saved using py2.x and are loaded by py3.x
# Calculate Maximum distance vector size
d = []
matrix_nan_perc = pd.DataFrame(index=['SSHA_35', 'ADG443_NN_OLCI_50km', 'TSM_NN_OLCI_150km', 'SST_150km',
       'ADG443_NN_OLCI_150km', 'TSM_NN_OLCI_5km', 'ADG443_NN_OLCI_5km',
       'KD490_M07_OLCI_5km', 'SST_5km', 'SST_50km', 'CHL_OC4ME_OLCI_50km',
       'TSM_NN_OLCI_50km', 'KD490_M07_OLCI_150km', 'CHL_OC4ME_OLCI_5km',
       'CHL_OC4ME_OLCI_150km', 'KD490_M07_OLCI_50km'])
for filename in npz_files:
    matrix, distance,_ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, filename))
    matrix_nan = matrix.isna()
    matrix_nan_N = matrix_nan.sum()/matrix_nan.shape[0]*100
    matrix_nan_perc = pd.concat([matrix_nan_perc, matrix_nan_N], axis=1)
    
    d.append(len(distance))
for item in matrix_nan_perc.index:
    # Compute greater than 50%
    gt_50 = matrix_nan_perc.loc[item] > 50
    gt_50_per = gt_50.sum()/gt_50.size*100
    
    plt.close(fig)
    fig = plt.figure(figsize=(15,10))
    ax = fig.gca()
    plt.hist(matrix_nan_perc.loc[item], bins=20)
#    plt.magnitude_spectrum(matrix)
#    plt.magnitude_spectrum(label)
    plt.xlabel('% of Missing Values', fontsize=18)
    plt.ylabel('# of tracks out of {0}'.format(matrix_nan_perc.shape[1]), fontsize=18)
    plt.title(item, fontsize=23)
    plt.text(0.2, 0.85, '{0:.2f}% of tracks have more than 50% of missing values'.format(gt_50_per), fontsize=15,
        bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10}, transform=ax.transAxes)
     
    plotpath = r'C:\Users\vlachos\Desktop\SSTlevel4'.replace('\\','\\')
    fig.savefig(os.path.join(plotpath, item+'.png'), dpi=300, bbox_inches='tight')
 