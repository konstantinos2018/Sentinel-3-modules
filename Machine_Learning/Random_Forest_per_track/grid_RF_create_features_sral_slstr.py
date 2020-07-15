# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import sys
import datetime as dt
from scipy import spatial
import scipy.signal as scsign
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
import pandas as pd
import time
from collections import OrderedDict
# My Modules
import ml_utilities
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty
import S3coordtran as s3ct
import s3utilities

# =============================================================================
# BEGIN
# =============================================================================
class ContinueI(Exception):
    pass

continue_i = ContinueI()

# =============================================================================
# GRID WITH QUERY POINTS
# =============================================================================
spatial_resolution = 330 # spatial resolution/pixel size of altimetry [meters]
sral_winsize = 35 # the window size that was used during the SSHA filtering
                  # in order to produce the features. 35 corresponds to ~11.55 km
                  # This will be the spatial resolution and pixel size of the grid
grid_step = 330*35 # Distance between points [meters]
x_min = -2750000 # x coordinate min [meters]
x_max = -1250000 # x coordinate max [meters]
y_min = 3800000 # y coordinate min [meters]
y_max = 4700000 # y coordinate max [meters]

x_query, y_query, n_x, n_y = ml_utilities.create_grid_true(x_min, x_max, y_min, y_max, grid_step)

del grid_step, spatial_resolution, sral_winsize, x_min, x_max, y_min, y_max

# =============================================================================
# BUILD FEATURE MATRIX ON THE QUERY POINTS
# =============================================================================
## Case One
#paths = {'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
#               'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
#               }
#
#common_date = {'OLCI': ['S3A_OL_2_WFR____20180630T143358_20180630T143658_20180701T201102_0179_033_039_2340_MAR_O_NT_002.SEN3', 'A'],
#         'SLSTR': ['S3A_SL_2_WST____20180630T014255_20180630T032354_20180701T102321_6059_033_031______MAR_O_NT_002.SEN3', 'B']
#         }
# Case Two
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }

# Read npz list
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr'.replace('\\','\\')
npz_files = os.listdir(path)
npz_files = [item for item in npz_files if 'npz' in item]

# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)
common_date = ml_utilities.products_from_npz(common_date, npz_files)

# Remove Dublicates
for key in common_date.keys():
    common_date[key] = list(OrderedDict.fromkeys(common_date[key]))

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1

lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']          
   
bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []

radius_slstr = {'SST_5km': 5000,
#                'SST_12.5km': 12500, 
#                'SST_16km': 16000,
#                'SST_32km': 32000,
                'SST_53km': 53000,
#                'SST_75km': 75000,
#                'SST_95km': 95000,
                'SST_105km': 105000,
#                'SST_125km': 125000,
                'SST_150km': 150000
                }

vec_len = []
# Dictionary which includes masked arrays that will then be saved as an npz
# and be used as CNN inputs
data_matrix = {}

total_iteration = 0
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        if f_sral[16:24] == f_slstr[16:24]:
            total_iteration = total_iteration + 1
counter = 0
k = 1
# Plot common dates
time_start = time.time()
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        if f_sral[16:24] == f_slstr[16:24]:
            pass
        else:
            continue
    #        if (f_slstr[16:24] == f_sral[16:24]) and (f_olci[16:24] == f_sral[16:24]):
    #            if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
    #                break
    #                if f_sral[:3] != 'S3B':
    #                    break
        sys.stdout.write("\rProgress... {0:.2f} %\n".format((counter/total_iteration)*100))
        sys.stdout.flush()
    
        # ========= SLSTR
        try:
            fname = os.listdir(os.path.join(paths['SLSTR'], f_slstr))
            fullpath = os.path.join(os.path.join(paths['SLSTR'], f_slstr), fname[0])
        except:
            # Keep name of file which was not opened correctly
            bad_slstr_1.append(f_slstr)
            continue
        # Read netcdf
        try:
            lonsl, latsl, varValues, l2p_flags, quality_level = ncman.slstr1D_read_nc(fullpath, lst_slstr)
        except:
            # Keep name of file which was not read correctly
            bad_slstr_2.append(f_slstr)
            continue
        
        # transform coordinates
        xsl, ysl = s3ct.slstr_olci_coordtran(lonsl, latsl, inEPSG, outEPSG)
        del lonsl, latsl
        # subset dataset
        varValues = ncman.slstr_olci_subset_nc(xsl, ysl, varValues, bound)
        # Extract bits of the l2p_flags flag
        flag_out = S3postproc.extract_bits(l2p_flags, 16)
        # Extract dictionary with flag meanings and values
        l2p_flags_mean, quality_level_mean = S3postproc.extract_maskmeanings(fullpath)
        # Create masks
        masks = S3postproc.extract_mask(l2p_flags_mean, flag_out, 16)
        del flag_out, l2p_flags_mean
        
        # Apply masks to given variables
        # Define variables separately
        sst = varValues['sea_surface_temperature']
        del varValues
        sst, outmask_sst = S3postproc.apply_masks_slstr(sst, 'sea_surface_temperature', masks, quality_level)
        del masks
        
        # Apply flag masks
        xsl = xsl[outmask_sst]
        ysl = ysl[outmask_sst]
        del outmask_sst
        # Apply varValues (e.g. sst) masks
        xsl = xsl[sst.mask]
        ysl = ysl[sst.mask]
        sst = sst.data[sst.mask] - 273 # convert to Celsius
              
        # Check if empty
        if check_npempty(sst):
            print('SLSTR date {0} is empty'.format(f_slstr[16:24]))
            total_iteration = total_iteration - 1
            continue
    #        if check_npempty(ssha):
    #            print('SSHA date {0} is empty'.format(f_sral[16:24]))
    #            total_iteration = total_iteration - 1
    #            continue
        
        # =============================================================================
        # INTERPOLATE AND FILTER SST            
        # =============================================================================
        # Interpolate IDW
        sst_interp = S3postproc.ckdnn_traject_idw(x_query, y_query, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
        
        # Check if empty
        if check_npempty(sst_interp):
            print('SST interpolated | date {0} is empty'.format(f_slstr[16:24]))
            total_iteration = total_iteration - 1
            continue
    #            # Check if all nan in sst interp
    #            if np.all(np.isnan(sst_interp)) == True:
    #                print('SST interpolated | date {0} only contains NaNs'.format(f_slstr[16:24]))
    #                continue
        try:
            num_iter_3 = 1
            for radius_key, radius_value in zip(radius_slstr.keys(), radius_slstr.values()):
                print('{0} SST radius out of {1}\n'.format(num_iter_3, len(radius_slstr)))
                
                # Low pass moving average filter
                data_matrix[radius_key] = S3postproc.twoDirregularFilter(x_query, y_query, sst_interp, xsl, ysl, sst, {'r':radius_value})
                # If some of the smoothed sst_interp version only contain NaNs, then continue to next file
                if np.all(np.isnan(data_matrix[radius_key])) == True:
                    print('SST estimate | date {0} only contains NaNs'.format(f_slstr[16:24]))
                    total_iteration = total_iteration - 1
                    raise continue_i
                # Calculate MASK and replace
                # =============================================================================
                # SST estimate OUTLIER DETECTION
                # =============================================================================
                num_iter_3 = num_iter_3 + 1
    #                # Choose inside percentiles
    #                Q1, Q3 = np.nanpercentile(data_matrix[radius_key],q=[25,75], interpolation='linear')
    #                IQR = Q3 - Q1 # Interquartile range
    #                thresh_low = Q1 - 1.5*IQR # lower outlier threshold
    #                thresh_up = Q3 + 1.5*IQR # upper outlier threshold
    #                idx_sst = (data_matrix[radius_key] > thresh_low) & (data_matrix[radius_key] < thresh_up)
                # INSERT each smoothed version of sst_est with MASK in the dictionary data_matrix
    #                data_matrix[radius_key] = np.ma.array(data_matrix[radius_key], mask=~idx_sst)
                data_matrix[radius_key] = np.ma.array(data_matrix[radius_key])
                
        except ContinueI:
            continue
        
        print('SST: OK')
        
    #        # ====== DISTANCE
    #        # Compute distance vector
    #        dst = S3postproc.sral_dist(xsr, ysr)
    #        # INSERT distance to data_matrix (no need for mask)
    #        data_matrix['Distance'] = dst
    #        print('DISTANCE: OK')
        # ======= X, Y COORDINATES
        data_matrix['X'] = x_query
        data_matrix['Y'] = y_query
        
        print('COORDINATES: OK')
        
        data_matrix['n_x'] = n_x
        data_matrix['n_y'] = n_y
        
        fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
        fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
        
        fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
        fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')        
        
        # INSERT sensors and dates of data
        data_matrix['Metadata'] = 'SRAL: {0}\nSLSTR: {1}'.format(f_sral[16:31], f_slstr[16:31])
        # path to save npz files
        save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\grid_npz_sral_slstr_perTrack_RF'.replace('\\', '\\')
        # npz general filename
        filename = '{0}_{1}__{2}'.format(f_sral[:3], fdate_sral, fdate_slstr)
        
    #        k = 1
        if os.path.exists(os.path.join(save_path, '{0}.npz'.format(filename))):
           np.savez_compressed(os.path.join(save_path, '{0}_{1}.npz'.format(filename, k)), data_matrix)
           k =+ 1
        else:
           np.savez_compressed(os.path.join(save_path, filename), data_matrix)
        
        counter = counter + 1

time_end = time.time()
print('The time taken is {0}'.format(time_end-time_start))
