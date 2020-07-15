# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:32:39 2019

@author: vlachos
"""
# =============================================================================
# DESCRIPTION:
# The following script makes a comparison between different 2D moving averaging
# filtering windows using the manual approach for the SLSTR.
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import os
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
import datetime as dt
import s3utilities
from scipy import spatial
import scipy.signal as scsign
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import pdb # Debugger module


# Find common dates
# Gulf Stream
paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }
# North Sea
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }

# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream
#bound = [3500000, 4300000, 3100000, 4000000] # North Sea

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
#                if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
#                    break
            #====================================== SRAL                    
            fullpath = os.path.join(os.path.join(paths['SRAL'], f_sral), fname_sral)
            # Read netcdf
            try:
                lonsr, latsr, ssha, flagsr = ncman.sral_read_nc(fullpath, lst_sral)
            except:
                bad_sral.append(f_sral)
                continue
            # transform coordinates
            xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
            
            # subset dataset
            xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)
            # Apply flags/masks
            ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
            
            # Apply outmask
            xsr = xsr[outmask]
            ysr =  ysr[outmask]
            # Clear workspace
            del lonsr, latsr, flagsr, outmask, ssha
            
            #=========================================== SLSTR
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
            
            fdate = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
            fdate = fdate.strftime('%Y-%m-%d %H_%M_%S')
            
             
            # Interpolate IDW
            sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
            
            # If interpolation fails for ALL points, then the go to next date
            if np.all(np.isnan(sst_interp)) == True:
                continue
            else:
                pass
#                    lst.append(np.where(sst_est[0] < 5001)[1].max() + 1)
#                    pdb.set_trace() # ENTER DEBUG MODE
#                    continue
#            print(np.sum(np.isnan(sst_interp)), sst_interp.size)
            # Compute distances between SRAL points
            dst = S3postproc.sral_dist(xsr, ysr)
#            pdb.set_trace()
            # Initialize dictionary with variables
            var = {}
            distance = {}
            legend_labels = []
            threshold = 3*330 # meters
            # =============================================================================
            #      FILTERS                    
            # =============================================================================
           
            radius_size = [5000, 12500, 16000, 32000, 53000] # search radius length [meters]
            radius_size.sort() # sort in increasing order
            labels = ['2d_MovAv_R'+str(item)+' m' for item in radius_size] # Create list of strings with names
            
            line_props = [
#                    {'color':'y', 'linestyle':'-', 'alpha':1, 'linewidth':1.3},
                    {'color':'g', 'linestyle':'-', 'alpha':0.6, 'linewidth':1.3},
                    {'color':'r', 'linestyle':'-', 'alpha':0.6, 'linewidth':2},
#                        {'color': '#ffcdc3', 'linestyle': '-', 'alpha':0.8, 'linewidth':3},
                    {'color':'k', 'linestyle':'-', 'alpha':0.6, 'linewidth':3},
                    {'color':'m', 'linestyle':'-', 'alpha':0.6, 'linewidth':3},
                    {'color':'c', 'linestyle':'-', 'alpha':0.6, 'linewidth':4}
            ]
            if True:
                radius_size.insert(0, 0) # insert 0 for Original ssha
                labels.insert(0, 'SST_interp') # Insert in the 0th element
                line_props.insert(0, {'color':'b', 'linestyle':'-', 'alpha':0.6, 'linewidth':0.8})
                
            for radius, lab, prop in zip(radius_size, labels, line_props):
                if lab == 'SST_interp':
                    var[lab] = [sst_interp, prop]
                    distance[lab] = dst
                    legend_labels.append(lab)
                else:
                    # 2d filter SST
                    sst_movAv = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':radius})
                    # Insert nans based on threshold value (distance between points)
                    distance[lab], sst_movAv_nan, _ = S3postproc.sral_dist_nans(dst, sst_movAv, threshold)
                    # Create variables 
                    var[lab] = [sst_movAv_nan,
                       prop]
                    # labels of the plot legend
                    legend_labels.append(lab)
                        
                    # Low pass moving average filter
#                    sst_movAv = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':radius_size})
#                # "Trend" moving average filter
##                sst_movAv = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':50000})
#                # Spatial detrend (sst_est = sst - sst_movAv)
##                sst_est = sst_movAvlow - sst_movAv
#                
#                # Interpolate k-NN
##                sst_est = S3postproc.ckdnn_traject_knn(xsr, ysr, xsl, ysl, sst, {'distance_upper_bound':10000/2, 'k': 10**4})
#
#                # Choose inside percentiles
##                idx = (ssha_m > np.percentile(ssha_m, 15)) & (ssha_m < np.percentile(ssha_m, 85))
##                ssha_m = ssha_m[idx]
#
#                
#                # Normalize variables between upper and lower bounds
##                ub = 1  # upper bound
##                lb = -1 # lower bound
#
#                # Rescale
##                ssha_m_nan = S3postproc.rescale_between(ssha_m_nan, ub, lb)
##                sst_est = S3postproc.rescale_between(sst_est, ub, lb)
#                
            # Convert labels from list to tuple
            legend_labels = tuple(legend_labels)
            # Plot
            S3plots.multiple_cross_sections(var, distance, legend_labels, fdate)
            