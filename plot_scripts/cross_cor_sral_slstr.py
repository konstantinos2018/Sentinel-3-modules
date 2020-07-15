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
import s3utilities
from scipy import spatial
import scipy.signal as scsign
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
import pdb
import pandas as pd
# My Modules
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty
import S3plots
import S3coordtran as s3ct

# Find common dates
#paths = {'SRAL': r'H:\MSc_Thesis_05082019\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'H:\MSc_Thesis_05082019\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'H:\MSc_Thesis_05082019\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }
# Gulf Stream Test
paths = {'SRAL': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }


# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']

lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    

bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []

# Plot common dates
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        if f_slstr[16:24] == f_sral[16:24]:                   
            # ========== SRAL                    
            fullpath = os.path.join(os.path.join(paths['SRAL'], f_sral), fname_sral)
            # Read netcdf
            try:
                lonsr, latsr, ssha, flagsr = ncman.sral_read_nc(fullpath, lst_sral)
            except:
                bad_sral.append(f_sral)
                continue
            # transform coordinates
            xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
            del lonsr, latsr
            # subset dataset
            xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)
            ssha = ssha['ssha_20_ku']
            # Apply flags/masks
            ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
            
            # Clear
            del flagsr
            
            # ========= SLSTR
            try:
                fname = os.listdir(os.path.join(paths['SLSTR'], f_slstr))
                fullpath = os.path.join(os.path.join(paths['SLSTR'], f_slstr), fname[0])
            except:
                bad_slstr_1.append(f_slstr)
                continue
            # Read netcdf
            try:
                lonsl, latsl, varValues, l2p_flags, quality_level = ncman.slstr1D_read_nc(fullpath, lst_slstr)
            except:
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
            sst, outmasks = S3postproc.apply_masks_slstr(sst, 'sea_surface_temperature', masks, quality_level)
            del masks
            
            # Apply flag masks
            xsl = xsl[outmasks]
            ysl = ysl[outmasks]
            del outmasks
            # Apply varValues (e.g. sst) masks
            xsl = xsl[sst.mask]
            ysl = ysl[sst.mask]
            sst = sst.data[sst.mask] - 273 # convert to Celsius
            
            # Check if empty
            if check_npempty(sst) or check_npempty(ssha):
                continue
            
            # Interpolate IDW
            sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
            
            # Check if empty
            if check_npempty(sst_interp):
                continue
            
            # Low pass moving average filter
            sst_movAv_low = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':50000})
            sst_movAv_vlow = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':150000})
            # Spatial Detrend
            sst_est = sst_movAv_low - sst_movAv_vlow
                       # Check if ALL interpolated SST are NaNs, If they are go to the next file
            if np.all(np.isnan(sst_est)) == True:
                continue
              
            # Choose inside percentiles
            idx = (ssha > np.percentile(ssha, 1)) & (ssha < np.percentile(ssha, 99))
            idx = idx & outmask
            
            # Keep ssha_m
            ssha_m_keep = np.ones_like(ssha) * ssha
            
            # ============= Filter
            log_window_size = []
            ssha[~idx] = np.nan
            window_size = 303
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
    
            # ====== 2nd filteer (larger window size)
            log_window_size2 = []
            ssha_m_keep[~idx] = np.nan
            window_size2 = 901
            # Check window size
            if ssha_m_keep.size < window_size2:
                window_size2 = ssha_m_keep.size
                # Check if window size is odd or even (needs to be odd)
                if window_size2 % 2 == 0:
                    window_size2 = window_size2 + 1
                # Log which files do not use the default window size
                log_window_size2.append(f_sral)
            ssha_m_keep = astro_conv(ssha_m_keep, np.ones((window_size2))/float(window_size2), boundary='extend',
                                nan_treatment='interpolate',preserve_nan=True)
            
            # Subtract large trend
            ssha_m = ssha_m - ssha_m_keep
            
            # =============================================================================
            # Impute missing values
            # =============================================================================
            # SRAL
            num_sample = ssha_m.size
            ssha_m = pd.DataFrame(ssha_m)
            ssha_m_fill = ssha_m.interpolate(method='linear',
#                                          limit=500,
                                          limit_direction='both')
            ssha_m_fill = np.asarray(ssha_m_fill).reshape(num_sample)
            
            # SLSTR
            sst_est = pd.DataFrame(sst_est)
            sst_est_fill = sst_est.interpolate(method='linear',
#                                          limit=500,
                                          limit_direction='both')
            sst_est_fill = np.asarray(sst_est_fill).reshape(num_sample)
            
            # =============================================================================
            # PLOT
            # =============================================================================
            print('ssha ', np.sum(np.isnan(ssha_m_fill)))
            print('sst ', np.sum(np.isnan(sst_est_fill)))
            
            fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
            fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
            
            fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
            fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
            
            dicinp = {'plttitle': 'SRAL ' + fdate_sral + '\n' + 'SLSTR ' + fdate_slstr,
                      'filename': fdate_sral + '__' + fdate_slstr}
            
            axis_labels = {'Y': 'SSHA [m]',
                           'X': 'SST [$^\circ$C]'}
            # Variables in dictionary
            variables = {'SRAL': ssha_m_fill,
                         'SLSTR': sst_est_fill,
                         'OLCI': []
                         }
            # Number of lag shifts
            if ssha.size < 400:
                num_lag = ssha.size - 1
            else:
                num_lag = 400
            
            # plot path
#            plotpath = r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Cross_correlations'.replace('\\','\\') # North Sea
            plotpath = r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Scatter\scatter_10'.replace('\\','\\') # Gulf Stream
#            plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Cross_Correlation'.replace('\\','\\') # Gulf stream
#            S3plots.cross_correl_2variable(variables, num_lag, dicinp, plotpath)
            S3plots.scatter_sral_slstr(variables['SRAL'], variables['SLSTR'],dicinp, axis_labels, plotpath)
            
            del sst, xsl, ysl
            
    del xsr, ysr, ssha_m