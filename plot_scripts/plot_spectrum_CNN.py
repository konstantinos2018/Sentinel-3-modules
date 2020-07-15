# -*- coding: utf-8 -*-

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import pandas as pd
import sys
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
import pdb
# My Modules
import ml_utilities

#global x_train
#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files\S3A_2018-05-25 14_54_00__2018-05-25 02_16_36.npz'.replace('\\','\\')
#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr\S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz'.replace('\\','\\')
models_path = r'C:\Users\vlachos\Desktop\SSTlevel4'.replace('\\','\\')
filespath = r'C:\Users\vlachos\Desktop\npz_files_sral_sstL4_1DCNN_real'.replace('\\','\\')
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
    if distance.size < 2000:
        continue
    # Imputate NaNs
    matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=False)
    
    label = np.array(matrix['SSHA_105'])
#    label = ml_utilities.matrix_min_max_rescale(label, 1, -1, axis=0)
    matrix = matrix.drop(columns=['SSHA_35', 'SSHA_71', 'SSHA_105'])
    matrix = ml_utilities.matrix_min_max_rescale(matrix, 0.5, -0.5, axis=0)
    matrix = np.array(matrix).squeeze()
#    _, ccorr, _, _ = plt.xcorr(label, matrix, usevlines=True, maxlags=400, normed=True, lw=2)
#    ccorr_max = ccorr.max()
#    ccorr_min = ccorr.min()
    
#    if ((ccorr_max < 0.5) and (ccorr_max > -0.5)) or ((ccorr_min < 0.5) and (ccorr_min > -0.5)):
#        continue
    
    fig = plt.figure(figsize=(15,10))
    plt.magnitude_spectrum(matrix)
    plt.magnitude_spectrum(label)
    plt.xlim(0, 0.05)
    plt.legend(['SSHA', 'SST'])
    plt.title(filename[:-4], fontsize=23)
    fig.savefig(os.path.join(r'C:\Users\vlachos\Desktop\SSTlevel4'.replace('\\','\\'), filename+ +'_autocorr'+'.png'), dpi=300, bbox_inches='tight')