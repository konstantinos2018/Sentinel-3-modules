# -*- coding: utf-8 -*-
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import matplotlib.pyplot as plt
import os
# My Modules
import ml_utilities

filename = 'xcorr_timeDifference.npz'
filespath = r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Cross_Correlation'.replace('\\','\\')

data = np.load(os.path.join(filespath, filename), encoding='latin1', allow_pickle=True)
# Retrieve dictionary
data = data['arr_0'].item()
#plt.close('all')
matrix, distance,_ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, filename))
#if distance.size < 2000:
#    continue
# Imputate NaNs
matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=False)

label = np.array(matrix['SSHA_105'])