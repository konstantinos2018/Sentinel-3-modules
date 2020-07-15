# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# SRAL - SLSTR
# =============================================================================
# =============================================================================
#  IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import datetime as dt
from scipy import spatial
import scipy.signal as scsign
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
import pdb
import pandas as pd
import warnings
import time
# My Modules
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty
import S3plots
import S3coordtran as s3ct
import s3utilities

# =============================================================================
# BEGIN
# =============================================================================
class ContinueI(Exception):
    pass
continue_i = ContinueI()

# Find common dates
## Gulf Stream Test
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
#         }
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }

# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream

bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']

lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    

fname_olci = 'sub_OLCI.nc'
lst_olci = ['ADG443_NN', 'CHL_OC4ME', 'KD490_M07', 'TSM_NN', 'WQSF']   

bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []
bad_olci = []

#windows_sral = [0, 11, 21, 35, 101, 201, 303, 333, 455, 901]
radius_slstr = {'SST_5km': 5000,
                'SST_50km': 50000,
                'SST_150km': 150000
                }
radius_olci = {'OLCI_5km': 5000,
                'OLCI_50km': 50000,
                'OLCI_150km': 150000
                }

vec_len = []
max_vec_len = 3127
# Dictionary which includes masked arrays that will then be saved as an npz
# and be used as CNN inputs
data_matrix = {}

# Count roughly how many iterations
total_iteration = 0
counter = 1
k = 1
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        for f_olci in common_date['OLCI']:
            if (f_slstr[16:24] == f_sral[16:24]) and (f_olci[16:24] == f_sral[16:24]):
                total_iteration = total_iteration + 1
                
time_start = time.time()  
# Plot common dates
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        for f_olci in common_date['OLCI']:
            try:
                if (f_slstr[16:24] == f_sral[16:24]) and (f_olci[16:24] == f_sral[16:24]):
        #            if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
        #                break
    #                if f_sral[:3] != 'S3B':
    #                    break
                    # ========== SRAL                    
                    fullpath = os.path.join(os.path.join(paths['SRAL'], f_sral), fname_sral)
                    # Read netcdf
                    try:
                        lonsr, latsr, ssha, flagsr = ncman.sral_read_nc(fullpath, lst_sral)
                    except:
                        # Keep name of file which was not read correctly
                        bad_sral.append(f_sral)
                        continue
                    # transform coordinates
                    xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
                    del lonsr, latsr
                    # subset dataset
                    xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)
                    ssha = ssha['ssha_20_ku']
                    # Apply flags/masks
                    _, outmask_ssha = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
                    
                    # Clear
                    del flagsr
                    
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
                        print 'SLSTR date {0} is empty'.format(f_slstr[16:24])
                        continue
                    if check_npempty(ssha):
                        print 'SSHA date {0} is empty'.format(f_sral[16:24])
                        continue
                    
                    # =============================================================================
                    # INTERPOLATE AND FILTER SST            
                    # =============================================================================
                    # Interpolate IDW
                    sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
                    
                    # Check if empty
                    if check_npempty(sst_interp):
                        print 'SST interpolated | date {0} is empty'.format(f_slstr[16:24])
                        continue
        #            # Check if all nan in sst interp
        #            if np.all(np.isnan(sst_interp)) == True:
        #                print 'SST interpolated | date {0} only contains NaNs'.format(f_slstr[16:24])
        #                continue
                    try:
                        for radius_key, radius_value in zip(radius_slstr.keys(), radius_slstr.values()):
                            
                            # Low pass moving average filter
                            data_matrix[radius_key] = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':radius_value})
                            # If some of the smoothed sst_interp version only contain NaNs, then continue to next file
                            if np.all(np.isnan(data_matrix[radius_key])) == True:
                                print 'SST estimate | date {0} only contains NaNs'.format(f_slstr[16:24])
                                raise continue_i
                            # Calculate MASK and replace
                            # =============================================================================
                            # SST estimate OUTLIER DETECTION                    
                            # =============================================================================
                            
                            # Choose inside percentiles
                            Q1, Q3 = np.nanpercentile(data_matrix[radius_key],q=[25,75], interpolation='linear')
                            IQR = Q3 - Q1 # Interquartile range
                            thresh_low = Q1 - 1.5*IQR # lower outlier threshold
                            thresh_up = Q3 + 1.5*IQR # upper outlier threshold
                            idx_sst = (data_matrix[radius_key] > thresh_low) & (data_matrix[radius_key] < thresh_up)
                            # INSERT each smoothed version of sst_est with MASK in the dictionary data_matrix
                            data_matrix[radius_key] = np.ma.array(data_matrix[radius_key], mask=~idx_sst)
                            
                    except ContinueI:
                        continue
                    
                    print 'SST: OK'
        #            sst_movAv_vlow = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':150000})
                    # Spatial Detrend
        #            sst_est = sst_movAv_low - sst_movAv_vlow
                  
        #            # Count the length of each SSHA vector size 
    #                vec_len.append(ssha.size)
    #                continue
                
                    # =============================================================================
                    # SSHA OUTLIER DETECTION            
                    # =============================================================================
                    # Choose inside percentiles
                    Q1, Q3 = np.nanpercentile(ssha,q=[25,75], interpolation='linear')
                    IQR = Q3 - Q1 # Interquartile range
                    thresh_low = Q1 - 1.5*IQR # lower outlier threshold
                    thresh_up = Q3 + 1.5*IQR # upper outlier threshold
                    
                    # Outlier mask
                    idx = (ssha > thresh_low) & (ssha < thresh_up)
                    # Outlier and flag mask
                    idx = idx & outmask_ssha
                    
        #            # Keep ssha_m
        #            ssha_m_keep = np.ones_like(ssha) * ssha
                    
                    # =============================================================================
                    # FILTER SSHA
                    # =============================================================================
                    log_window_size = []
                    ssha[~idx] = np.nan
                    window_size = 35
                    # Check window size
                    if ssha.size < window_size:
                        window_size = ssha.size
                        # Check if window size is odd or even (needs to be odd)
                        if window_size % 2 == 0:
                            window_size = window_size + 1
                        # Log which files do not use the default window size
                        log_window_size.append(f_sral)
                    ssha_m = astro_conv(ssha, np.ones((window_size))/float(window_size), boundary='extend',
                                        nan_treatment='interpolate', preserve_nan=True)
                    
                    # INSERT ssha with MASK in the dictionary data_matrix
                    # Basically, the ~idx AFTER the smoothing represents NaNs, because
                    # ~idx positions have been replaced by NaNs
                    data_matrix['SSHA_'+ str(window_size)] = np.ma.array(ssha_m, mask=~idx)
                    
                    print 'SSHA: OK'
    
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
                                print 'OLCI date {0} is empty'.format(f_olci[16:24])
                                total_iteration = total_iteration - 1
                                raise continue_i
                            # Interpolate IDW
                            olci_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xol[olci_outmasks][olci.mask], yol[olci_outmasks][olci.mask], olci.data[olci.mask], {'k':12, 'distance_upper_bound':330*np.sqrt(2)})
                            
                            # Check if empty
                            if check_npempty(olci_interp):
                                print 'OLCI interpolated | date {0} is empty'.format(f_olci[16:24])
                                total_iteration = total_iteration - 1
                                raise continue_i
    
                            num_iter_2 = 1
                            for radius_key, radius_value in zip(radius_olci.keys(), radius_olci.values()):
                                dt_m_key = variable_name + '_' + radius_key
                                # Low pass moving average filter
                                data_matrix[dt_m_key] = S3postproc.twoDirregularFilter(xsr, ysr, olci_interp, xol[olci_outmasks][olci.mask], yol[olci_outmasks][olci.mask], olci.data[olci.mask], {'r':radius_value})
                                # If some of the smoothed olci_interp version only contain NaNs, then continue to next file
                                if np.all(np.isnan(data_matrix[dt_m_key])) == True:
                                    print 'OLCI estimate | date {0} only contains NaNs'.format(f_olci[16:24])
                                    total_iteration = total_iteration - 1
                                    raise continue_i
                                print '{0} OLCI radius out of 3\n'.format(num_iter_2)
                                num_iter_2 = num_iter_2 + 1
                                # Calculate MASK and replace
                                # =============================================================================
                                # SST estimate OUTLIER DETECTION                    
                                # =============================================================================
                                
                                # Choose inside percentiles
                                Q1, Q3 = np.nanpercentile(data_matrix[dt_m_key],q=[25,75], interpolation='linear')
                                IQR = Q3 - Q1 # Interquartile range
                                thresh_low = Q1 - 1.5*IQR # lower outlier threshold
                                thresh_up = Q3 + 1.5*IQR # upper outlier threshold
                                idx_olci = (data_matrix[dt_m_key] > thresh_low) & (data_matrix[dt_m_key] < thresh_up)
                                # INSERT each smoothed version of sst_est with MASK in the dictionary data_matrix
                                data_matrix[dt_m_key] = np.ma.array(data_matrix[dt_m_key], mask=~idx_olci)
                            
                            print '{0} OLCI variable out of 4\n'.format(num_iter_1)
                            num_iter_1 = num_iter_1 + 1
                                
                            del olci_outmasks
                            
                    except ContinueI:
                        continue
                    
                    print 'OLCI: OK'
                    
                    
                    # Compute distance vector
                    dst = S3postproc.sral_dist(xsr, ysr)
                    # INSERT distance to data_matrix (no need for mask)
                    data_matrix['Distance'] = dst
                    print 'DISTANCE: OK'
                    
                    fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
                    fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
                    
                    fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
                    fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
                    
                    # INSERT sensors and dates of data
                    data_matrix['Metadata'] = 'SRAL: {0}\nSLSTR: {1}'.format(f_sral[16:31], f_slstr[16:31])
                    # path to save npz files
                    save_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_1DCNN_real'.replace('\\', '\\')
                    # npz general filename
                    filename = '{0}_{1}__{2}'.format(f_sral[:3], fdate_sral, fdate_slstr)
                    
    #                k = 1
                    if os.path.exists(os.path.join(save_path, '{0}.npz'.format(filename))):
                       np.savez_compressed(os.path.join(save_path, '{0}_({1}).npz'.format(filename, k)), data_matrix)
                       k = k + 1
                    else:
                       np.savez_compressed(os.path.join(save_path, filename), data_matrix)
            except:
                continue
#            pdb.set_trace()



# Plot histogram regarding the length of each SSHA vector. I did this in order
# to decide whether I'm going to kick out the vectors of size < than the largest
# smoothing window size (in my case ~901 window size)
#vec_len = np.array(vec_len)
#min_len = vec_len.min()
#max_len = vec_len.max()
#median_len = int(np.median(vec_len))
#plt.figure(figsize=(10, 8))
#plt.hist(vec_len, bins=int(round(vec_len.size**(1/3.0)*2)))
#plt.xlabel('Length of track', fontsize=18)
#plt.ylabel('Total # of tracks', fontsize=18)
#plt.title('SSHA vector length', fontsize=23)
#plt.text(500, 400,'min = {0}\nmax = {1}\nmedian = {2}\n# tracks = {3}'.format(min_len, max_len, median_len, vec_len.size),
#         fontsize=13, bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})                   
                  