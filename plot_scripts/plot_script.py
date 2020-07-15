# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:37:32 2019

@author: vlachos
"""
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
import pdb
# My Modules
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty
import S3plots
import S3coordtran as s3ct
import s3utilities

type_script = 'SCATTER'
# Find common dates
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
         }
## Mediterranean Test
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\SLSTR'.replace('\\','\\')
#         }
## Gulf Stream Test
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
#         }


# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)


# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
#outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1

bound = [3500000, 4300000, 3100000, 4000000] # North Sea
#bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']

#fname_olci = 'sub_OLCI.nc'
lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    

bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []       

www = [455, 333, 333]
rrr = [53000, 53000, 32000]
ppp = [r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Scatter\scatter_7'.replace('\\','\\'),
       r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Scatter\scatter_8'.replace('\\','\\'),
       r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Scatter\scatter_9'.replace('\\','\\')]
# for ww, rr, pp in zip(www, rrr, ppp):
# Plot common dates
for f_sral in common_date['SRAL']:
    for f_slstr in common_date['SLSTR']:
        if f_slstr[16:24] == f_sral[16:24]:                   
                    
                    # SRAL                    
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
                    # Apply flags/masks
                    ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
                    # Clear
                    del flagsr
                    
                    # Apply outmask
                    xsr = xsr[outmask]
                    ysr =  ysr[outmask]
                    del outmask
                    
                    fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
                    fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
                    
                    
                    # SLSTR
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
                    if check_npempty(sst) or check_npempty(ssha_m):
                        continue
                    
                    fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
                    fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
                    
                    dicinp = {'plttitle': 'SRAL ' + fdate_sral + '\n' + 'SLSTR ' + fdate_slstr,
                              'filename': fdate_sral + '__' + fdate_slstr}
                    
                    if type_script == 'MAP':
                        # Compute percentiles
                        idx = (ssha_m > np.percentile(ssha_m, 5)) & (ssha_m < np.percentile(ssha_m, 95))
                        ssha_m = ssha_m[idx]
                        # Filter SRAL
                        ssha_m = scsign.medfilt(ssha_m, 47)
                        
                        # Plot map
                        plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR'.replace('//','//')
                        S3plots.slstr_sral_trajplot(xsr, ysr, ssha_m, xsl, ysl, sst, shppath, bound, dicinp, plotpath)
                        
                    elif type_script == 'SCATTER' or 'HISTOGRAM':
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
                        # Choose inside percentiles
                        idx = (ssha_m > np.percentile(ssha_m, 20)) & (ssha_m < np.percentile(ssha_m, 80))
                        
                        # Keep ssha_m
                        ssha_m_keep = np.ones_like(ssha_m) * ssha_m
                        
                        # ============= Filter
                        log_window_size = []
                        ssha_m[~idx] = np.nan
                        window_size = 303
                        # Check window size
                        if ssha_m.size < window_size:
                            window_size = ssha_m.size
                            # Check if window size is odd or even (needs to be odd)
                            if window_size % 2 == 0:
                                window_size = window_size + 1
                            # Log which files do not use the default window size
                            log_window_size.append(f_sral)
                        ssha_m = astro_conv(ssha_m, np.ones((window_size))/float(window_size), boundary='extend',
                                            nan_treatment='interpolate',preserve_nan=True)

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
                    
                        # Check if ALL interpolated SST are NaNs, If they are go to the next file
                        if np.all(np.isnan(sst_est)) == True:
                            continue
                        
                        if type_script == 'SCATTER':
                            # Give nan values based on index
#                            ssha_m[idx] = np.nan
                            
                            # Plot scatter
                            axis_labels = {'Y': 'SSHA [m]',
                                           'X': 'SST [$^\circ$C]'}
                            # plot path
                            plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Scatter'.replace('\\','\\')
                            
                            S3plots.scatter_sral_slstr(ssha_m, sst_est, dicinp, axis_labels, plotpath)
                            
                        elif type_script == 'HISTOGRAM':
                            # Plot SRAL histogram
                            dicinp = {'plttitle': 'SRAL ' + fdate_sral,
                                      'filename': fdate_sral}
                            axis_labels = {'X': 'SSHA [m]',
                                           'Y': '# Observations'}

                            S3plots.histogram_sral(ssha_m, dicinp, axis_labels)

                    del sst, xsl, ysl
    del xsr, ysr, ssha_m
#%% 
# =============================================================================
# SRAL - OLCI
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
#from netCDF4 import Dataset
#import matplotlib.pyplot as plt
import os
import datetime as dt
import pdb
import numpy as np
import scipy.signal as scsign
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
import s3utilities

type_script = 'MAP'

## Find common dates
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }
# Gulf Stream
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }

# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)


# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

# Bounding box region coordinates
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream
#bound = [3500000, 4300000, 3100000, 4000000] # North Sea

#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']

fname_olci = 'sub_OLCI.nc'
lst_olci = ['CHL_OC4ME', 'WQSF']    

bad_sral = []
bad_olci = []
      
# Plot common dates
for f_sral in common_date['SRAL']:
    for f_olci in common_date['OLCI']:
        if f_olci[16:24] == f_sral[16:24]:                   
                    
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
                    # Apply flags/masks
                    ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
                    # Clear
                    del flagsr
                    
                    # Apply outmask
                    xsr = xsr[outmask]
                    ysr =  ysr[outmask]
                    del outmask
                    
                    fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
                    fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
                    
                    
                    # OLCI
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
                    # Clearn
                    del flagol
                    # Extract dictionary with flag meanings and values
                    bitval = S3postproc.extract_maskmeanings(fullpath)
                    # Create masks
                    masks = S3postproc.extract_mask(bitval, flag_out, 64)
                    # clean
                    del flag_out, bitval
                    # Apply masks to given variables
                    # Define variables separately
                    chl_oc4me = varValues['CHL_OC4ME']
                    del varValues
                    chl_oc4me, outmasks = S3postproc.apply_masks_olci(chl_oc4me, 'CHL_OC4ME', masks)
                    # Clean
                    del masks
                    
                    # Apply flag masks
                    xol = xol[outmasks]
                    yol = yol[outmasks]
                    del outmasks
                    
                    # Apply chl_nn mask
                    xol = xol[chl_oc4me.mask]
                    yol = yol[chl_oc4me.mask]
                    chl_oc4me = chl_oc4me.data[chl_oc4me.mask]
                    
                    fdate_olci = dt.datetime.strptime(f_olci[16:31], '%Y%m%dT%H%M%S')
                    fdate_olci = fdate_olci.strftime('%Y-%m-%d %H_%M_%S')
                    
                    dicinp = {'plttitle': 'SRAL ' + fdate_sral + '\n' + 'OLCI ' + fdate_olci,
                              'filename': fdate_sral + '__' + fdate_olci}
                    
                    if type_script == 'MAP':
                        # Compute percentiles
                        idx = (ssha_m > np.percentile(ssha_m, 5)) & (ssha_m < np.percentile(ssha_m, 95))
                        ssha_m = ssha_m[idx]
                        # Filter SRAL
                        ssha_m = scsign.medfilt(ssha_m, 47)
                        # Plot map
                        S3plots.olci_sral_trajplot(xsr, ysr, ssha_m, xol, yol, chl_oc4me, shppath, bound, dicinp)
                        
#%%
# =============================================================================
# SRAL - SLSTR Plot each SRAL with each SLSTR in order to check if there
# are correlations that have a time delay
# =============================================================================
# =============================================================================
# IMPORTS                        
# =============================================================================
# Python Modules
#from netCDF4 import Dataset
#import matplotlib.pyplot as plt
import numpy as np
import os
import datetime as dt
from scipy import spatial
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
import s3utilities

type_script = 'SCATTER'
# Find common dates
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }
# Gulf Stream
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }


# Folder names with the common dates
path_sral = os.listdir(paths['SRAL'])
path_sral = [f for f in path_sral if f[:3] == 'S3A']
path_slstr = os.listdir(paths['SLSTR'])
path_slstr = [f for f in path_slstr if f[:3] == 'S3A']


# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\')
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']

#fname_olci = 'sub_OLCI.nc'
lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    

bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []

#path_sral = path_sral[1:]
# Plot common dates
for f_sral in path_sral:
    for f_slstr in path_slstr:
        # Compute time difference (days) between products
        t_diff = np.abs(dt.datetime.strptime(f_sral[16:24], '%Y%m%d') - 
                        dt.datetime.strptime(f_slstr[16:24], '%Y%m%d'))
        # If the time difference between SRAL and SLSTR is less than 3 months (90 days)
        # plot the pair SRAL-SLSTR
        if t_diff <= dt.timedelta(days=90):
            # SRAL                    
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
            # Apply flags/masks
            ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
            # Clear
            del flagsr
            
            # Apply outmask
            xsr = xsr[outmask]
            ysr =  ysr[outmask]
            del outmask
            
            fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
            fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
            
            
            # SLSTR
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
            
            fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
            fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
            
            dicinp = {'plttitle': 'SRAL ' + fdate_sral + '\n' + 'SLSTR ' + fdate_slstr,
                      'filename': fdate_sral + '__' + fdate_slstr}
            
    
            if type_script == 'MAP':
                # Plot map
                S3plots.slstr_sral_trajplot(xsr, ysr, ssha_m, xsl, ysl, sst, shppath, bound, dicinp)
            elif type_script == 'SCATTER' or 'HISTOGRAM':
                # Stack x and y coordinates of SRAL and variable
                xysralstack = np.dstack([xsr, ysr])[0]
                xyslstack = np.dstack([xsl, ysl])[0]
                # Compute nearest neighbors
                tree = spatial.cKDTree(xyslstack)
                out = tree.query(xysralstack, k=1, distance_upper_bound=300)
                # find inf positions
                index_inf = np.isinf(out[0])
                # Interpolate using the kNN with k=1 and radius 300m
                out[1][index_inf] = out[1][index_inf] - 1
                sst_est = sst[out[1]]
                # Put NaNs where interpolation didn't work bacause points where too far
                sst_est[index_inf] =  np.nan
                
                # Check if ALL interpolated SST are NaNs, If they are go to the next file
                if np.logical_not(np.isnan(sst_est)).sum() == 0:
                    continue
                
                if type_script == 'SCATTER':
                    # Compute percentiles
                    perc_15 = ssha_m > np.percentile(ssha_m, 5)
                    perc_85 = ssha_m < np.percentile(ssha_m, 95)
                    # Mask data based on percentiles
                    ssha_m = ssha_m[perc_15 & perc_85]
                    sst_est = sst_est[perc_15 & perc_85]
            
                    # Plot scatter
                    axis_labels = {'X': 'SSHA [m]',
                               'Y': 'SST [Celcius]'}
                    S3plots.scatter_sral_slstr(ssha_m, sst_est, dicinp, axis_labels)
                elif type_script == 'HISTOGRAM':
                    
                    # Plot SRAL histogram
                    dicinp = {'plttitle': 'SRAL ' + fdate_sral,
                              'filename': fdate_sral}
                    axis_labels = {'X': 'SSHA [m]',
                                   'Y': '# Observations'}
                    # Compute percentiles
                    perc_15 = ssha_m > np.percentile(ssha_m, 5)
                    perc_85 = ssha_m < np.percentile(ssha_m, 95)
                    ssha_m = ssha_m[perc_15 & perc_85]
                    S3plots.histogram_sral(ssha_m, dicinp, axis_labels)
    
            del sst, xsl, ysl
    del xsr, ysr, ssha_m
