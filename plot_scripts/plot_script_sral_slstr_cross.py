# -*- coding: utf-8 -*-

# =============================================================================
# SRAL - SLSTR Trajectory
# =============================================================================
# =============================================================================
# SRAL - SLSTR
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
import scipy.signal as scsign
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import pdb
from astropy.convolution import convolve as astro_conv
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
from S3postproc import check_npempty
import s3utilities

def my_fun():

    filter_method = {'MEDIAN': False,
                     'AVERAGE': False,
                     'BUTTER': False,
                     'ASTROPY': True
                     }
    # Find common dates
#    paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#             'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#             'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#             }
    # Gulf Stream Test
    paths = {'SRAL': r'N:\My Documents\My Bulletin\SRAL'.replace('\\', '\\'),
             'OLCI': r'N:\My Documents\My Bulletin\OLCI'.replace('\\', '\\'),
             'SLSTR': r'N:\My Documents\My Bulletin\SLSTR'.replace('\\','\\')
             }

    
    # Folder names with the common dates
    common_date = s3utilities.find_common_dates(paths)
    
    
    # Define constants
    inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
    outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#    outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
   
#    bound = [3500000, 4300000, 3100000, 4000000] # North Sea
    bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Strea
    
    fname_sral = 'sub_enhanced_measurement.nc'
    lst_sral = ['ssha_20_ku', 'flags']
    
    #fname_olci = 'sub_OLCI.nc'
    lst_slstr = ['sea_surface_temperature', 'l2p_flags', 'quality_level']    
    
    bad_sral = []
    bad_slstr_1 = []
    bad_slstr_2 = []
    log_window_size = []
    log_window_size2 = []
    
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
                
                fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
                fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
                
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
                if check_npempty(sst):
                    continue
                fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
                fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H_%M_%S')
                
                
                dicinp = {'plttitle': 'SRAL ' + f_sral[:3] + ' ' + fdate_sral + '\n' + 'SLSTR ' + f_slstr[:3] + ' ' + fdate_slstr,
                          'filename': fdate_sral + '__' + fdate_slstr}
                
                # Interpolate IDW
                sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
                # Check if empty
                if check_npempty(sst_interp):
                    continue
                # Low pass moving average filter
                sst_movAvlow = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':50000})
                # "Trend" moving average filter
                sst_movAv = S3postproc.twoDirregularFilter(xsr, ysr, sst_interp, xsl, ysl, sst, {'r':150000})
                # Spatial detrend (sst_est = sst - sst_movAv)
                sst_est = sst_movAvlow - sst_movAv
                
                # Interpolate k-NN
#                sst_est = S3postproc.ckdnn_traject_knn(xsr, ysr, xsl, ysl, sst, {'distance_upper_bound':10000/2, 'k': 10**4})
                # If interpolation fails for ALL points, then the go to next date
                if np.all(np.isnan(sst_est)) == True:
                    continue
                else:
                    pass
#                    lst.append(np.where(sst_est[0] < 5001)[1].max() + 1)
#                    pdb.set_trace() # ENTER DEBUG MODE
#                    continue
                # Choose inside percentiles
                idx = (ssha_m > np.percentile(ssha_m,1)) & (ssha_m < np.percentile(ssha_m, 99))
                
                # Keep ssha_m
                ssha_m_keep = np.ones_like(ssha_m) * ssha_m
                        
                # Compute distances between SRAL points
#                dst_sr = S3postproc.sral_dist(xsr[idx], ysr[idx])
                dst_sr = S3postproc.sral_dist(xsr, ysr)
                dst_sl = S3postproc.sral_dist(xsr, ysr)
                

#                    # ================ Insert NaNs ===============
                # Choose filtering method
                if filter_method['ASTROPY'] == True:
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
                    
                    # ====== 2nd filter (larger window size)
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
                    
                elif filter_method['MEDIAN'] == True:
                    ssha_m = ssha_m[idx]
                    window_size = 35
                    # Insert nans
                    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold=3*340)
                    # Filter SSHA
                    ssha_m_nan = scsign.medfilt(ssha_m_nan, window_size)
                    
                elif filter_method['AVERAGE'] == True:
                    ssha_m = ssha_m[idx]
                    window_size = 35
                    # Moving Average filter SSHA
                    # Check window size
                    if ssha_m.size < window_size:
                        window_size = ssha_m.size
                        # Check if window size is odd or even (needs to be odd)
                        if window_size % 2 == 0:
                            window_size = window_size + 1
                        # Log which files do not use the default window size
                        log_window_size.append(f_sral)
                    ssha_m = np.convolve(ssha_m, np.ones((window_size))/window_size, mode='same')

                    # Insert nans
                    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr[idx], ssha_m, threshold=3*340)

                elif filter_method['BUTTER'] == True:
                    ssha_m = ssha_m[idx]
                    def butter_lowpass(cutoff, fs, order=5):
                        nyq = 0.5 * fs
                        normal_cutoff = cutoff / nyq
                        b, a = butter(order, normal_cutoff, btype='low', analog=False)
                        return b, a

                    def butter_lowpass_filter(data, cutoff, fs, order=5):
                        b, a = butter_lowpass(cutoff, fs, order=order)
                        y = lfilter(b, a, data)
                        return y
                    
                    # Filter requirements.
                    order = 3
                    fs = 8       # sample rate, Hz
                    cutoff = 0.25#2.667  # desired cutoff frequency of the filter, Hz                    
                    # Get the filter coefficients so we can check its frequency response.
                    b, a = butter_lowpass(cutoff, fs, order)                    
                    # Plot the frequency response.
#                        w, h = freqz(b, a, worN=8000)
#                        plt.subplot(2, 1, 1)
#                        plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#                        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#                        plt.axvline(cutoff, color='k')
#                        plt.xlim(0, 0.5*fs)
#                        plt.title("Lowpass Filter Frequency Response")
#                        plt.xlabel('Frequency [Hz]')
#                        plt.grid()
                    # Filter the data, and plot both the original and filtered signals.
                    ssha_m = butter_lowpass_filter(ssha_m, cutoff, fs, order)
                    # Insert nans
                    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold=3*340)
                
#                    dst_sl_nan, sst_est_nan, idx_sl_nan = S3postproc.sral_dist_nans(dst_sl, sst_est, threshold_sl)
                # Normalize variables between upper and lower bounds
                ub = 1  # upper bound
                lb = -1 # lower bound

                # Rescale
                ssha_m_nan = S3postproc.rescale_between(ssha_m, ub, lb)
                sst_est = S3postproc.rescale_between(sst_est, ub, lb)
                
                # Variables in dictionary
                variables = {'SRAL': ssha_m_nan,
                             'SLSTR': sst_est,
                             'OLCI': []
                             }
                distance = {'SRAL': dst_sr,
                            'SLSTR': dst_sl,
                            'OLCI': []
                            }
                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Trajectories'.replace('\\','\\') # Gulf stream
#                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Trajectories'.replace('\\','\\') # North Sea
                # Plot
                S3plots.sral_cross_sections(variables, distance, dicinp, plotpath)
                
    return bad_sral, bad_slstr_1, bad_slstr_2

if __name__ == '__main__':
    lst1, lst2, lst3 = my_fun()