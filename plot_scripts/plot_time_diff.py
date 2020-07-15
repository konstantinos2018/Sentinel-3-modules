# -*- coding: utf-8 -*-
# =============================================================================
# SRAL - SLSTR
# =============================================================================
# Import packages
import numpy as np
import os
import nc_manipul as ncman
import S3postproc
import S3plots
from S3postproc import check_npempty
import datetime as dt
import s3utilities
import matplotlib.pyplot as plt
import pdb
import S3coordtran as s3ct

# Gulf Stream Test
paths = {'SRAL': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
         'OLCI': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\', '\\'),
         'SLSTR': r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
         }
## North Sea
#paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }

# Folder names with the common dates
common_date = s3utilities.find_common_dates(paths)

# Define constants
inEPSG = inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
#    outEPSG = 'epsg:23031'# balearic sea (Mediterranean)
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1# Gulf Stream 1
#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
#    bound = [292779.99035029, 657943.31318846, 3986589.56737893, 4629517.01159] # Mediterranean_Test
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['time_20_ku', 'ssha_20_ku', 'flags']

#fname_olci = 'sub_OLCI.nc'
lst_slstr = ['time', 'sst_dtime', 'sea_surface_temperature']    

bad_sral = []
bad_slstr_1 = []
bad_slstr_2 = []
log_window_size = []
log_window_size2 = []
min_time_vec = []
max_time_vec = []

# open txt file
#txt_path = os.path.join(r'N:\My Documents\My Bulletin'.replace('\\','\\'), 'GulfStream_Dates.txt') # Gulf Stream
txt_path = os.path.join(r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Time_differences'.replace('\\','\\'), 'GulfStream_Dates.txt') # North Sea
#f_obj = open(txt_path, 'w')

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
                lonsr, latsr, var_sral, flagsr = ncman.sral_read_nc(fullpath, lst_sral)
            except:
                bad_sral.append(f_sral)
                continue
            
            # transform coordinates
            xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
            del lonsr, latsr
            # subset dataset
            xsr, ysr, var_sral, _ = ncman.sral_subset_nc(xsr, ysr, var_sral, flagsr, bound)
            
            ssha_time = var_sral['time_20_ku']
            ssha = var_sral['ssha_20_ku']
            
#                # Apply flags/masks
#                ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
            
#                # Apply outmask
#                xsr = xsr[outmask]
#                ysr =  ysr[outmask]
            # Clear workspace
            
            fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
            fdate_sral = fdate_sral.strftime('%Y-%m-%d %H:%M:%S')
            
            #=========================================== SLSTR
            try:
                fname = os.listdir(os.path.join(paths['SLSTR'], f_slstr))
                fullpath = os.path.join(os.path.join(paths['SLSTR'], f_slstr), fname[0])
            except:
                bad_slstr_1.append(f_slstr)
                continue
            # Read netcdf
            try:
                lonsl, latsl, varValues, _, _ = ncman.slstr1D_read_nc(fullpath, lst_slstr)
            except:
                bad_slstr_2.append(f_slstr)
                continue
        
            # transform coordinates
            xsl, ysl = s3ct.slstr_olci_coordtran(lonsl, latsl, inEPSG, outEPSG)
            del lonsl, latsl
            # subset dataset
            varValues = ncman.slstr_olci_subset_nc(xsl, ysl, varValues, bound)
                
#                # Extract bits of the l2p_flags flag
#                flag_out = S3postproc.extract_bits(l2p_flags, 16)
#                # Extract dictionary with flag meanings and values
#                l2p_flags_mean, quality_level_mean = S3postproc.extract_maskmeanings(fullpath)
#                # Create masks
#                masks = S3postproc.extract_mask(l2p_flags_mean, flag_out, 16)
#                del flag_out, l2p_flags_mean
            
            # Apply masks to given variables
            # Define variables separately
            sst_dtime = varValues['sst_dtime'].data[varValues['sst_dtime'].mask]
            sst_time_ref = varValues['time']
            
#            del varValues
            
            # Check if arrays are empty
            if check_npempty(ssha_time) or check_npempty(sst_dtime):
                continue

            
            fdate_slstr = dt.datetime.strptime(f_slstr[16:31], '%Y%m%dT%H%M%S')
            fdate_slstr = fdate_slstr.strftime('%Y-%m-%d %H:%M:%S')
#            
#            
#            dicinp = {'plttitle': 'SRAL ' + fdate_sral + '\n' + 'SLSTR ' + fdate_slstr,
#                      'filename': fdate_sral + '__' + fdate_slstr}
            
#                # Interpolate IDW
#                sst_interp = S3postproc.ckdnn_traject_idw(xsr, ysr, xsl, ysl, sst, {'k':12, 'distance_upper_bound':1000*np.sqrt(2)})
#                # Check if empty
#                if check_npempty(sst_interp):
#                    continue
#
#                if np.all(np.isnan(sst_est)) == True:
#                    continue
#                else:
#                    pass
#                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SLSTR\Trajectories'.replace('\\','\\') # Gulf stream
##                plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_SLSTR\Trajectories'.replace('\\','\\') # North Sea
#                # Plot
#                S3plots.sral_cross_sections(variables, distance, dicinp, plotpath)
            
            # Compute times
            # --- SRAL
            min_time_sral = ssha_time.min()
            max_time_sral = ssha_time.max()
            # How many seconds from 1970 to 2000 (Unix to Gregorian difference)
            unix_greg = dt.datetime(2000, 1, 1) - dt.datetime(1970, 1, 1)
            
            # --- SLSTR
            min_time_slstr = sst_time_ref + sst_dtime.min() # seconds after 2000-01-01T00:00:00.000
            max_time_slstr = sst_time_ref + sst_dtime.max()
            # How many seconds from 1970 to 2000 (Unix to ... difference)
            unix_slstr = dt.datetime(1981, 1, 1) - dt.datetime(1970, 1, 1)        

#            write_string = 'SRAL | {0} {1} | min time: {2} | max time: {3}\nSLSTR | {4} {5} | min time: {6} | max time: {7}\n\n'.format(f_sral[:3],
#                                   fdate_sral,
#                                   dt.datetime.fromtimestamp(min_time_sral + unix_greg.total_seconds()),
#                                   dt.datetime.fromtimestamp(max_time_sral + unix_greg.total_seconds()),
#                                   f_slstr[:3],
#                                   fdate_slstr,
#                                   dt.datetime.fromtimestamp(min_time_slstr + unix_slstr.total_seconds()),
#                                   dt.datetime.fromtimestamp(max_time_slstr + unix_slstr.total_seconds()))
#            f_obj.write(write_string)
            
            min_time_vec.append(np.abs(min_time_sral + unix_greg.total_seconds() - min_time_slstr - unix_slstr.total_seconds()))
            max_time_vec.append(np.abs(max_time_sral + unix_greg.total_seconds() - max_time_slstr - unix_slstr.total_seconds()))
# Close txt file
#f_obj.close()

min_time_vec = np.asarray(min_time_vec).squeeze()/60.0
max_time_vec = np.asarray(max_time_vec).squeeze()/60.0

fig = plt.figure(figsize=(15, 8))
font = {'size' : 18}
plt.rc('font', **font)
plt.hist(min_time_vec, bins=15)
plt.xlabel('time [min]', fontsize=18)
plt.ylabel('counts [#]', fontsize=18)
plt.title('Differences between time of arrival of SRAL/SLSTR', fontsize=23)