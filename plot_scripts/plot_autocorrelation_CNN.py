# -*- coding: utf-8 -*-

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
import pandas.plotting
# My Modules
import ml_utilities

#global x_train

#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files\S3A_2018-05-25 14_54_00__2018-05-25 02_16_36.npz'.replace('\\','\\')
#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr\S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz'.replace('\\','\\')
models_path = r"C:\Users\vlachos\Desktop\SSTlevel4".replace('\\','\\')
filespath = r'C:\Users\vlachos\Desktop\npz_files_sral_sstL4_1DCNN_real'.replace('\\','\\')
npz_files = os.listdir(filespath)

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
    if distance.size < 2000:
        continue
    # Imputate NaNs
    matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=False)
    
    label = matrix['SSHA_105']
#    label = ml_utilities.matrix_min_max_rescale(label, 1, -1, axis=0)
    matrix = matrix.drop(columns=['SSHA_35', 'SSHA_71', 'SSHA_105'])
#    matrix = ml_utilities.matrix_min_max_rescale(matrix, 0.5, -0.5, axis=0)
    fig = plt.figure(figsize=(15,10))
#    ax = fig.gca()
    pandas.plotting.autocorrelation_plot(label, label='SSHA')
    ax = pandas.plotting.autocorrelation_plot(matrix, label='SST L4')
    
    handles, labels = ax.get_legend_handles_labels()
#    plt.plot(distance, label, distance, matrix)
#    plt.acorr(matrix, usevlines=True, maxlags=400, normed=True, lw=2)
#    plt.xlim(0, 0.05)
    plt.legend(handles=handles, labels=labels, fontsize=20)
    plt.title(filename[:-4], fontsize=23)
    
    plotpath = r'C:\Users\vlachos\Desktop\SSTlevel4'.replace('\\','\\')
    fig.savefig(os.path.join(plotpath, filename +'_autocorr'+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)