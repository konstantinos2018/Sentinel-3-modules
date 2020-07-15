# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
from netCDF4 import Dataset
import datetime as dt
import os
import sys
import numpy as np
from astropy.convolution import convolve as astro_conv
import matplotlib.pyplot as plt
# My Modules
import S3coordtran as s3ct
import nc_manipul as ncman
import S3postproc
from S3postproc import check_npempty

# =============================================================================
# BEGIN
# =============================================================================
sst_level4_path = r'H:\MSc_Thesis_05082019\Data\SST_Level_4\aggregate__ghrsst_JPL_OUROCEAN-L4UHfnd-GLOB-G1SST_OI.ncml.nc'.replace('\\','\\')

sral_path = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\')
sral_files = os.listdir(sral_path)

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1

bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1

# Open netCDF
nc = Dataset(sst_level4_path)

lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij') # create meshgrid
rows, cols = lon_grid.shape
lon_grid = np.matrix.flatten(lon_grid) # convert to array
lat_grid = np.matrix.flatten(lat_grid) # convert to array
# transform coordinates
x_sst, y_sst = s3ct.sral_coordtran(lon_grid, lat_grid, inEPSG, outEPSG)
x_sst = np.reshape(x_sst, (rows, cols))
y_sst = np.reshape(y_sst, (rows, cols))
del lon, lat, lon_grid, lat_grid

#sst_tseries = nc.variables['analysed_sst'][:] - 273.15
#time = nc.variables['time'][:] # seconds since 1981-01-01 00:00:00

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']
bad_sral = []

counter = 0
total_iteration = 0
k = 1
# Dictionary which includes masked arrays that will then be saved as an npz
# and be used as CNN inputs
data_matrix = {}

#for f_sral in sral_files:
#    for i in range(nc.variables['time'].shape[0]):
#        f_sst_time = nc.variables['time'][i] # seconds since 1981-01-01 00:00:00
#        f_sst_time = dt.datetime.fromtimestamp(f_sst_time + dt.datetime(1981,1,1).timestamp()-dt.timedelta(hours=1).total_seconds())
#        f_sst_time = f_sst_time.strftime('%Y%m%d')
#        
#        if (f_sst_time == f_sral[16:24]):
#            total_iteration = total_iteration + 1
        
# Plot common dates
for f_sral in sral_files:
    for i in range(nc.variables['time'].shape[0]):
        f_sst_time = nc.variables['time'][i] # seconds since 1981-01-01 00:00:00
        f_sst_time = dt.datetime.fromtimestamp(f_sst_time + dt.datetime(1981,1,1).timestamp()-dt.timedelta(hours=1).total_seconds())
#        if (f_sst_time < dt.datetime(2019, 1, 26)) or (f_sral[:3] == 'S3A'):
#            continue
        f_sst_time = f_sst_time.strftime('%Y%m%d')
        
        if (f_sst_time == f_sral[16:24]):
#            if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
#                break
#                if f_sral[:3] != 'S3B':
#                    break
            print("File: {0} out of {1}\n".format(counter, total_iteration))
            
            # ========== SRAL                    
            fullpath = os.path.join(os.path.join(sral_path, f_sral), fname_sral)
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
            # Check if empty
            if check_npempty(ssha):
                print('SSHA | date {0} is empty'.format(f_sst_time))
                total_iteration = total_iteration - 1
                continue
            
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
            window_size = [35]
            for ws in window_size:
                # Check window size
                if ssha.size < ws:
                    ws = ssha.size
                    # Check if window size is odd or even (needs to be odd)
                    if ws % 2 == 0:
                        ws = ws + 1
                    # Log which files do not use the default window size
                    log_window_size.append(f_sral)
                else:
                    pass
                ssha_m = astro_conv(ssha, np.ones((ws))/float(ws), boundary='extend',
                                    nan_treatment='interpolate', preserve_nan=True)
                
                # INSERT ssha with MASK in the dictionary data_matrix
                # Basically, the ~idx AFTER the smoothing represents NaNs, because
                # ~idx positions have been replaced by NaNs
                data_matrix['SSHA_'+ str(ws)] = np.ma.array(ssha_m, mask=~idx)
            
            print('SSHA: OK')
            
            # -------- SST Level 4
            sst_level4 = nc.variables['analysed_sst'][i] # - 273.15
            sst_level4_mask = nc.variables['mask'][i] == 1 # Read Sea mask
            sst_level4 = sst_level4[sst_level4_mask] - 273.15# Apply mask and convert to Celcius
#            sst_level4 = np.matrix.flatten(sst_level4) # make array
            x_sst_temp = x_sst[sst_level4_mask] # Apply mask
            y_sst_temp = y_sst[sst_level4_mask] # Apply mask
            
            del sst_level4_mask
            # =============================================================================
            # INTERPOLATE           
            # =============================================================================
            # Interpolate IDW
            sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, x_sst_temp, y_sst_temp, sst_level4, {'k':12, 'distance_upper_bound':1002*np.sqrt(2)})
            
            # Check if empty
            if check_npempty(sst_interp):
                print('SST interpolated | date {0} is empty'.format(f_sst_time))
                total_iteration = total_iteration - 1
                continue
                
            data_matrix['SST_LEVEL4_1km'] = sst_interp
            # Low pass moving average filter
            try:
                data_matrix['SST_LEVEL4_10km'] = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, x_sst_temp, y_sst_temp, sst_level4, {'r':5000})
            except:
                continue
            
            print('SST LEVEL4: OK')
            # ====== DISTANCE
            # Compute distance vector
            dst = S3postproc.sral_dist(xsr, ysr)
            # INSERT distance to data_matrix (no need for mask)
            data_matrix['Distance'] = dst
            print('DISTANCE: OK')
            
            fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
            fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
            
            f_sst_time = dt.datetime.strptime(f_sst_time, '%Y%m%d')
            f_sst_time = f_sst_time.strftime('%Y-%m-%d')
            
            # INSERT sensors and dates of data
            data_matrix['Metadata'] = 'SRAL: {0}\nSST_Level4: {1}'.format(f_sral[16:31], f_sst_time)
                
            save_path = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\npz_files_sral_sstL4_1DCNN_real_October'.replace('\\', '\\')
            # npz general filename
            filename = '{0}_{1}__{2}'.format(f_sral[:3], fdate_sral, f_sst_time)
            
#            k = 1
            if os.path.exists(os.path.join(save_path, '{0}.npz'.format(filename))):
               np.savez_compressed(os.path.join(save_path, '{0}_({1}).npz'.format(filename, k)), data_matrix)
               k =+ 1
            else:
               np.savez_compressed(os.path.join(save_path, filename), data_matrix)
            
            del sst_level4, x_sst_temp, y_sst_temp, f_sst_time, sst_interp, ssha_m, ssha
            
            counter = counter + 1