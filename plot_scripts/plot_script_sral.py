# -*- coding: utf-8 -*-

# =============================================================================
# SRAL
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import datetime as dt
import numpy as np
import pdb
import scipy.stats as scst
import scipy.signal as scsign
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct

#'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\') # North Sea
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\','\\') # Gulf Stream
folder = os.listdir(path)
fname = 'sub_enhanced_measurement.nc'

# Plot and check which data are useful
lst = ['ssha_20_ku', 'flags']

inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1

bad = []
no_data = []

for f in folder:
    fullpath = os.path.join(os.path.join(path, f), fname)
    # Read netcdf
    try:
        lonsr, latsr, ssha, flagsr = ncman.sral_read_nc(fullpath, lst)
    except:
        bad.append(f)
        continue
    # transform coordinates
    xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
    del lonsr, latsr
    # subset dataset
    xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)    
    # Apply flags/masks
    ssha_m, outmask = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
    # Clear
    del flagsr, ssha
    
    # Apply outmask
    xsr = xsr[outmask]
    ysr =  ysr[outmask]
    del outmask
    
    fdate = dt.datetime.strptime(f[16:31], '%Y%m%dT%H%M%S')
    fdate = fdate.strftime('%Y-%m-%d %H_%M_%S')
    
    if np.all(np.isnan(ssha_m)) == True:
#        pdb.set_trace()
        no_data.append(f)
        continue
    
    # --- Filter ---
    # Compute percentiles
    idx = (ssha_m > np.percentile(ssha_m, 5)) & (ssha_m < np.percentile(ssha_m, 95))
    ssha_m = ssha_m[idx]
    # Filter SRAL
    ssha_m_filt = scsign.medfilt(ssha_m, 35)
    
    plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL'.replace('\\','\\')                 
    # Plot SRAL
#    S3plots.sral_scatter(xsr, ysr, ssha_m_filt, shppath, bound,'SSHA ' + fdate, plotpath)
    
    del ssha_m, xsr, ysr
    