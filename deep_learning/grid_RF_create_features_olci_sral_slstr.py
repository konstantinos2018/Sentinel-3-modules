# -*- coding: utf-8 -*-

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
import pandas as pd
import time
import sys
# My Modules
import ml_utilities
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty
import S3coordtran as s3ct

# =============================================================================
#  BEGIN
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
x_min = -2250000 # x coordinate min [meters]
x_max = -1250000 # x coordinate max [meters]
y_min = 4100000 # y coordinate min [meters]
y_max = 4600000 # y coordinate max [meters]

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
paths = {'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\','\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }

common_date = {'OLCI' : ['S3A_OL_2_WFR____20180807T144855_20180807T145155_20180808T222906_0180_034_196_2340_MAR_O_NT_002.SEN3'],
                'SLSTR' : ['S3A_SL_2_WST____20180807T015748_20180807T033847_20180808T120201_6059_034_188______MAR_O_NT_002.SEN3']
                }

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    

fname_olci = 'sub_OLCI.nc'
lst_olci = ['ADG443_NN', 'CHL_OC4ME', 'KD490_M07', 'TSM_NN', 'WQSF']        
   
bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []
bad_olci = []

#windows_sral = [0, 11, 21, 35, 101, 201, 303, 333, 455, 901]
radius_slstr = {'SST_5km': 5000,
                'SST_12.5km': 12500, 
                'SST_32km': 32000,
                'SST_53km': 53000,
                'SST_95km': 95000,
                'SST_125km': 125000,
                'SST_150km': 150000
                }

radius_olci = {'OLCI_5km': 5000,
                'OLCI_12.5km': 12500, 
                'OLCI_32km': 32000,
                'OLCI_95km': 95000,
                'OLCI_150km': 150000
                }

vec_len = []
# Dictionary which includes masked arrays that will then be saved as an npz
# and be used as CNN inputs
data_matrix = {}

total_iteration = 0
counter = 0
k = 1
# Plot common dates
time_start = time.time()
for f_slstr in common_date['SLSTR']:
    for f_olci in common_date['OLCI']:
#        if (f_slstr[16:24] == f_sral[16:24]) and (f_olci[16:24] == f_sral[16:24]):
#            if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
#                break
#                if f_sral[:3] != 'S3B':
#                    break
#        sys.stdout.write("\rProgress... {0:.2f} %\n".format((counter/total_iteration)*100))
#        sys.stdout.flush()

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
                print('{0} SST radius out of 8\n'.format(num_iter_3))
                num_iter_3 += 1
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
#            sst_movAv_vlow = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':150000})
        # Spatial Detrend
#            sst_est = sst_movAv_low - sst_movAv_vlow
      
#            # Count the length of each SSHA vector size 
#            vec_len.append(ssha.size)
#            continue
    
#        # =============================================================================
#        # SSHA OUTLIER DETECTION            
#        # =============================================================================
#        # Choose inside percentiles
#        Q1, Q3 = np.nanpercentile(ssha,q=[25,75], interpolation='linear')
#        IQR = Q3 - Q1 # Interquartile range
#        thresh_low = Q1 - 1.5*IQR # lower outlier threshold
#        thresh_up = Q3 + 1.5*IQR # upper outlier threshold
#        
#        # Outlier mask
#        idx = (ssha > thresh_low) & (ssha < thresh_up)
#        # Outlier and flag mask
#        idx = idx & outmask_ssha
#        
##            # Keep ssha_m
##            ssha_m_keep = np.ones_like(ssha) * ssha
#        
#        # =============================================================================
#        # FILTER SSHA
#        # =============================================================================
#        log_window_size = []
#        ssha[~idx] = np.nan
#        window_size = 35
#        # Check window size
#        if ssha.size < window_size:
#            window_size = ssha.size
#            # Check if window size is odd or even (needs to be odd)
#            if window_size % 2 == 0:
#                window_size = window_size + 1
#            # Log which files do not use the default window size
#            log_window_size.append(f_sral)
#        ssha_m = astro_conv(ssha, np.ones((window_size))/window_size, boundary='extend',
#                            nan_treatment='interpolate', preserve_nan=True)
#        
#        # INSERT ssha with MASK in the dictionary data_matrix
#        # Basically, the ~idx AFTER the smoothing represents NaNs, because
#        # ~idx positions have been replaced by NaNs
#        data_matrix['SSHA_'+ str(window_size)] = np.ma.array(ssha_m, mask=~idx)
#        
#        print('SSHA: OK')

        # ================================ OLCI
        fullpath = os.path.join(os.path.join(paths['OLCI'], f_olci), fname_olci)
        # Read netcdf
        try:
            lonol, latol, varValues, flagol = ncman.olci1D_read_nc(fullpath, lst_olci)
        except:
            bad_olci.append(f_olci)
            continue
        
        # transform coordinates
        xol, yol = s3ct.slstr_olci_coordtran(lonol.data, latol.data, inEPSG, outEPSG)
        del lonol, latol
        # subset dataset
        varValues = ncman.slstr_olci_subset_nc(xol, yol, varValues, bound)
        # Apply flags/masks
        # Extract bits of the wqsf flag
        flag_out = S3postproc.extract_bits(flagol.data, 64)
        # Clear
        del flagol
        # Extract dictionary with flag meanings and values
        bitval = S3postproc.extract_maskmeanings(fullpath)
        # Create masks
        masks = S3postproc.extract_mask(bitval, flag_out, 64)
        # clean
        del flag_out, bitval
        # Apply masks to given variables
        try:
            num_iter_1 = 1
            for variable_name in lst_olci[:-1]:    
                # Define variables separately          
                olci = varValues[variable_name]
#                        pdb.set_trace()
#                        del varValues
                olci, olci_outmasks = S3postproc.apply_masks_olci(olci, variable_name, masks)
#                        pdb.set_trace()
                # Clean
#                        del masks
                
#                        # Apply flag masks
#                        xol = xol[olci_outmasks]
#                        yol = yol[olci_outmasks]
#                        del olci_outmasks
                
#                        # Apply chl_nn mask
#                        xol = xol[olci.mask]
#                        yol = yol[olci.mask]
#                        olci = olci.data[olci.mask]
                
                # Check if olci variables are empty
                if check_npempty(olci):
                    print('OLCI date {0} is empty'.format(f_olci[16:24]))
                    total_iteration = total_iteration - 1
                    raise continue_i
                # Interpolate IDW
                olci_interp = S3postproc.ckdnn_traject_idw(x_query, y_query, xol[olci_outmasks][olci.mask], yol[olci_outmasks][olci.mask], olci.data[olci.mask], {'k':12, 'distance_upper_bound':330*np.sqrt(2)})
                
                # Check if empty
                if check_npempty(olci_interp):
                    print('OLCI interpolated | date {0} is empty'.format(f_olci[16:24]))
                    total_iteration = total_iteration - 1
                    raise continue_i

                num_iter_2 = 1
                for radius_key, radius_value in zip(radius_olci.keys(), radius_olci.values()):
                    dt_m_key = variable_name + '_' + radius_key
                    # Low pass moving average filter
                    data_matrix[dt_m_key] = S3postproc.twoDirregularFilter(x_query, y_query, olci_interp, xol[olci_outmasks][olci.mask], yol[olci_outmasks][olci.mask], olci.data[olci.mask], {'r':radius_value})
                    # If some of the smoothed olci_interp version only contain NaNs, then continue to next file
                    if np.all(np.isnan(data_matrix[dt_m_key])) == True:
                        print('OLCI estimate | date {0} only contains NaNs'.format(f_olci[16:24]))
                        total_iteration = total_iteration - 1
                        raise continue_i
                    print('{0} OLCI radius out of 5\n'.format(num_iter_2))
                    num_iter_2 += 1
                    # Calculate MASK and replace
                    # =============================================================================
                    # OLCI estimate OUTLIER DETECTION                    
                    # =============================================================================
                    
#                    # Choose inside percentiles
#                    Q1, Q3 = np.nanpercentile(data_matrix[dt_m_key],q=[25,75], interpolation='linear')
#                    IQR = Q3 - Q1 # Interquartile range
#                    thresh_low = Q1 - 1.5*IQR # lower outlier threshold
#                    thresh_up = Q3 + 1.5*IQR # upper outlier threshold
#                    idx_olci = (data_matrix[dt_m_key] > thresh_low) & (data_matrix[dt_m_key] < thresh_up)
                    # INSERT each smoothed version of sst_est with MASK in the dictionary data_matrix
#                    data_matrix[dt_m_key] = np.ma.array(data_matrix[dt_m_key], mask=~idx_olci)
                    data_matrix[dt_m_key] = np.ma.array(data_matrix[dt_m_key])
                
                print('{0} OLCI variable out of 4\n'.format(num_iter_1))
                num_iter_1 += 1
                    
                del olci_outmasks
                
        except ContinueI:
            continue
        
        print('OLCI: OK')
        
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
        
#        fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
#        fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
        
        fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
        fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
        
        fdate_olci = dt.datetime.strptime(f_olci[16:31], '%Y%m%dT%H%M%S')
        fdate_olci = fdate_olci.strftime('%Y-%m-%d %H_%M_%S')
        
        
        # INSERT sensors and dates of data
#        data_matrix['Metadata'] = 'SRAL: {0}\nSLSTR: {1}\nOLCI: {2}'.format(f_sral[16:31], f_slstr[16:31], f_olci[16:31])
        data_matrix['Metadata'] = 'SLSTR: {0}\nOLCI: {1}'.format(f_slstr[16:31], f_olci[16:31])
        # path to save npz files
#        save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files'.replace('\\', '\\')
        save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\grid_npz_sral_slstr_olci_RF\RF_complete_dataset_model_sral_slstr_olci'.replace('\\', '\\')
        # npz general filename
#        filename = '{0}_{1}__{2}'.format(f_sral[:3], fdate_sral, fdate_slstr)
        filename = '{0}_{1}__{2}'.format(f_slstr[:3], fdate_slstr, fdate_olci)
        
#        k = 1
        if os.path.exists(os.path.join(save_path, '{0}.npz'.format(filename))):
           np.savez_compressed(os.path.join(save_path, '{0}_{1}.npz'.format(filename, k)), data_matrix)
           k =+ 1
        else:
           np.savez_compressed(os.path.join(save_path, filename), data_matrix)
        
        counter = counter + 1
#                pdb.set_trace()
time_end = time.time()
print('The time taken is {0}'.format(time_end-time_start))
