# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# =============================================================================
# SRAL - SLSTR
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import datetime as dt
from scipy import spatial
import scipy.signal as scsign
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
import sys
import pandas as pd
# My Modules
import nc_manipul as ncman
from S3postproc import check_npempty
import S3plots
import S3coordtran as s3ct
import ml_utilities


# Gulf Stream Test
filespath = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\npz_files_sral_sstL4_1DCNN_real_October'.replace('\\','\\')
npz_files = os.listdir(filespath)
npz_files = [item for item in npz_files if '.npz' in item]
cross_corr_max = []
cross_corr_min = []
cross_corr_magn_max = []
N_npz_files = len(npz_files)

# Plot common dates
for n_file, filename in enumerate(npz_files):
    
        # Progress
#        print('{0} out of {1}\n'.format(n_file, N_npz_files))
        sys.stdout.write('\rFiles: {0} out of {1}'.format(n_file, N_npz_files))
        sys.stdout.flush()
        
        matrix, distance ,_ = ml_utilities.feature_matrix_from_npz(os.path.join(filespath, filename))

        if distance.size < 2000:
            continue
        # Imputate NaNs
        matrix, _ = ml_utilities.imputate_nans_feature_matrix(matrix, method='Interpolate', drop_nan=False)
#        print('hello')
        label = np.array(matrix['SSHA_35'])
#        matrix = matrix.drop(columns=['SSHA_35', 'SSHA_303', 'SST_LEVEL4_1km'])
        matrix = matrix['SST_LEVEL4_10km']
        matrix = np.array(matrix).squeeze()
        _, ccc, _, _ = plt.xcorr(label, matrix, usevlines=True, maxlags=400, normed=True, lw=2)
        cross_corr_max.append(ccc.max())
        cross_corr_min.append(ccc.min())
#        cross_corr_magn_max.append(np.abs(ccc).max()) # keep max of magnitude of cross-corr
        
#        continue
        fdate_sral = filename[4:23]
#        fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
        
        fdate_sst_l4 = filename[25:35]
#        fdate_sst_l4 = fdate_sst_l4.strftime('%Y-%m-%d %H_%M_%S')
        
        dicinp = {'plttitle': 'SSHA {0}\nSST L4 {1}'.format(fdate_sral,fdate_sst_l4),
                  'filename': '{0}__{1}'.format(fdate_sral, fdate_sst_l4)}
        
        axis_labels = {'Y': 'SSHA [m]',
                       'X': 'SST L4 10km [$^\circ$C]'}
        # Variables in dictionary
        variables = {'SRAL': label,
                     'SLSTR': matrix,
                     'OLCI': []
                     }
        # Number of desired lag shifts
        num_lag = 400
        # Change lag shift if the array size is large than the
        # default
        if label.size < 400:
            num_lag = label.size
        
        
        # plot path
        plotpath = r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SST_LEVEL4\Cross_correlations_SSTL4_SSHA_October'.replace('\\','\\') # Gulf stream
#        S3plots.cross_correl_2variable(variables, num_lag, dicinp, plotpath)
        
        del label, matrix, distance
        
        plt.close('all')

# Convert data into nparrays
cross_corr_min = np.array(cross_corr_min)
cross_corr_max = np.array(cross_corr_max)
#ccorr_data = pd.DataFrame([cross_corr_min, cross_corr_max]).transpose()
#ccorr_data_max = ccorr_data.max(axis=1)
# Keep the max positive and min negatives
max_pos = cross_corr_max[cross_corr_max > 0]
min_neg = cross_corr_min[cross_corr_min < 0]

# Plot minimum and maximum normalized cross correlation values
font = {'size' : 18}
plt.rc('font', **font)
plt.figure(figsize=(10, 8))
plt.hist(cross_corr_max, bins=15, color='b', alpha=0.6)
plt.hist(cross_corr_min, bins=15, fill=False, edgecolor='black')
plt.xlabel('Normalized Cross-Correlations', fontsize=18)
plt.ylabel('# of counts', fontsize=18)
plt.title('SSHA-SST L4\ntracks >2000 observations\nmax lag = $\pm$400', fontsize=23)
plt.legend(['Max cross-corrs', 'Min cross-corrs'], fontsize=16, loc='best')
#plt.text(0.20,1000 , 'Features:\nSST versions', fontsize=13,
#         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

# Plot maximum positive and minimum negative cross-correlations
plt.figure(figsize=(10, 8))
plt.hist(max_pos, bins=15, color='b', alpha=0.6)
plt.hist(min_neg, bins=15, fill=False, edgecolor='black')
plt.xlabel('Normalized Cross-Correlations', fontsize=18)
plt.ylabel('# of counts', fontsize=18)
plt.title('SSHA-SST L4\ntracks >2000 observations\nmax lag = $\pm$400', fontsize=23)
plt.legend(['Max Positive cross-corrs', 'Min Negative cross-corrs'], fontsize=16, loc='best')
#plt.text(0.20,1000 , 'Features:\nSST versions', fontsize=13,
#         bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

