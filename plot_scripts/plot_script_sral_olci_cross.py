# =============================================================================
# SRAL - OLCI Trajectory
# =============================================================================
# =============================================================================
# SRAL - OLCI
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
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
import pdb
import sys
import time
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
import s3utilities
from S3postproc import check_npempty

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
    paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
             'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
             'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
             }
    
    
    # Folder names with the common dates
    common_date = s3utilities.find_common_dates(paths)
    
    # Define constants
    inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
    outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#    outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#    bound = [3500000, 4300000, 3100000, 4000000] # North Sea
    bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream
    
    fname_sral = 'sub_enhanced_measurement.nc'
    lst_sral = ['ssha_20_ku', 'flags']
    
    fname_olci = 'sub_OLCI.nc'
    lst_olci = ['ADG443_NN', 'WQSF']        
    
    bad_sral = []
    bad_olci = []       
    log_window_size = []
    log_window_size2 = []
    counter = 0
    n =  len(paths['SRAL'])*len(paths['OLCI'])
    
    # Plot common dates
    for f_sral in common_date['SRAL']:
        for f_olci in common_date['OLCI']:
            if f_olci[16:24] == f_sral[16:24]:
#                if (dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') < dt.datetime(2018, 6, 28)) or (dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') > dt.datetime(2018, 8, 3)):
#                    pass
#                else:
#                    n = n - 1
#                    continue
                # === Progress
                sys.stdout.write("\rProgress... {0:.2f}%".format((float(counter)/n)*100))
                sys.stdout.flush()
                
                #================================= SRAL                    
                
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
                ssha = ssha['ssha_20_ku']
                # Apply flags/masks
                ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
                
                # Apply outmask
                xsr = xsr[outmask]
                ysr =  ysr[outmask]
                # Clear workspace
                del lonsr, latsr, flagsr, outmask, ssha
                
                fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
                fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')
                
                
                #================================ OLCI
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
                # Define variables separately                
                chl_oc4me = varValues['ADG443_NN']
                del varValues
                chl_oc4me, outmasks = S3postproc.apply_masks_olci(chl_oc4me, 'ADG443_NN', masks)
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
                if check_npempty(chl_oc4me):
                    continue
                
                fdate_olci = dt.datetime.strptime(f_olci[16:31], '%Y%m%dT%H%M%S')
                fdate_olci = fdate_olci.strftime('%Y-%m-%d %H_%M_%S')
                
                dicinp = {'plttitle': 'SRAL ' + f_sral[:3] + ' ' + fdate_sral + '\n' + 'OLCI ' + f_olci[:3] + ' ' + fdate_olci,
                          'filename': fdate_sral + '__' + fdate_olci}
                 
                # Interpolate IDW
                olci_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xol, yol, chl_oc4me, {'k':12, 'distance_upper_bound':330*np.sqrt(2)})
                # Check if empty
                if check_npempty(olci_interp):
                    continue
                # Low pass moving average filter
                olci_movAvlow = S3postproc.twoDirregularFilter(xsr, ysr, olci_interp, xol, yol, chl_oc4me, {'r':50000})
                # "Trend" moving average filter
                olci_movAv = S3postproc.twoDirregularFilter(xsr, ysr, olci_interp, xol, yol, chl_oc4me, {'r':150000})
                # Spatial detrend (sst_est = sst - sst_movAv)
                olci_est = olci_movAvlow - olci_movAv
                
                # If interpolation fails for ALL points, then go to next date
                if np.all(np.isnan(olci_est)) == True:
                    continue
                else:
                    pass

                # Choose inside percentiles
                idx = (ssha_m > np.percentile(ssha_m,1)) & (ssha_m < np.percentile(ssha_m, 99))
                
                # Keep ssha_m
                ssha_m_keep = np.ones_like(ssha_m) * ssha_m
                
                # Compute distances between SRAL points
#                dst_sr = S3postproc.sral_dist(xsr[idx], ysr[idx])
                dst_sr = S3postproc.sral_dist(xsr, ysr)
                dst_ol = S3postproc.sral_dist(xsr, ysr)
                

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
                    
                # Choosle filtering method
                if filter_method['MEDIAN'] == True:
                    # Insert nans
                    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold_sr)
                    # Filter SSHA
                    ssha_m_nan = scsign.medfilt(ssha_m_nan, 31)
                    
                elif filter_method['AVERAGE'] == True:
                    # Moving Average filter SSHA
                    ssha_m = np.convolve(ssha_m, np.ones((29))/ssha_m.size, mode='same')
                    sst_est = np.convolve(olci_est, np.ones((29))/olci_est.size, mode='same')
                    # Insert nans
                    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold_sr)
                    
                elif filter_method['BUTTER'] == True:
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
#                dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold_sr)
                    
#                    dst_sl_nan, sst_est_nan, idx_sl_nan = S3postproc.sral_dist_nans(dst_sl, sst_est, threshold_sl)
                # Normalize variables between upper and lower bounds
                ub = 1  # upper bound
                lb = -1 # lower bound

                # Rescale
                ssha_m_nan = S3postproc.rescale_between(ssha_m, ub, lb)
                olci_est = S3postproc.rescale_between(olci_est, ub, lb)
                
                # Variables in dictionary
                variables = {'SRAL': ssha_m_nan,
                             'SLSTR': [],
                             'OLCI': olci_est
                             }
                distance = {'SRAL': dst_sr,
                            'SLSTR': [],
                            'OLCI': dst_ol
                            }
                
                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_OLCI\Trajectories'.replace('\\','\\') # Gulf stream
#                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_OLCI\Trajectories'.replace('\\','\\') # North Sea
                # Plot
                S3plots.sral_cross_sections_olci(variables, distance, dicinp, plotpath)
                
                counter = counter + 1
                
                del olci_est, olci_interp, olci_movAv, olci_movAvlow, ssha_m, ssha_m_nan, ssha_m_keep, 
    return bad_sral, bad_olci

if __name__ == '__main__':
    t_start = time.time()
    bad_sral, bad_olci = my_fun()
    t_end = time.time()
    print('It took {0} secs to run'.format(t_end-t_start))