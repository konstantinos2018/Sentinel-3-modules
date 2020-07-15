# -*- coding: utf-8 -*-

# =============================================================================
# OLCI
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
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct

#path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\') # North Sea
path = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\OLCI'.replace('\\','\\') # Gulf Stream 1

folder = os.listdir(path)
fname = 'sub_OLCI.nc'

# Plot and check which data are useful
lst = ['CHL_OC4ME', 'WQSF']

inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
shppath = r'H:\MSc_Thesis_05082019\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1

bad = []
no_data = []

for f in folder:
#    if f[16:24] == '20180602':
#        pass
#    else:
#        continue
    
    fullpath = os.path.join(os.path.join(path, f), fname)
    # Read netcdf
    try:
        lonol, latol, varValues, flagol = ncman.olci1D_read_nc(fullpath, lst)
    except:
        bad.append(f)
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
    del flag_out
    # Apply masks to given variables
    # Define variables separately
    chl_oc4me = varValues['CHL_OC4ME']
    chl_oc4me, outmasks = S3postproc.apply_masks_olci(chl_oc4me, 'CHL_OC4ME', masks)
    # Clean
    del masks
    
    # Apply flag masks
    xol = xol[outmasks]
    yol = yol[outmasks]
    
    # Apply chl_nn mask
    xol = xol[chl_oc4me.mask]
    yol = yol[chl_oc4me.mask]
    chl_oc4me = chl_oc4me.data[chl_oc4me.mask]
    
    fdate = dt.datetime.strptime(f[16:31], '%Y%m%dT%H%M%S')
    fdate = fdate.strftime('%Y-%m-%d %H_%M_%S')
    
    if np.all(np.isnan(chl_oc4me)) == True:
#        pdb.set_trace()
        no_data.append(f)
        continue
    
    plotpath = r'H:\MSc_Thesis_05082019\Data\Satellite\Outputs\Gulf_Stream_1\OLCI\chl_oc4me'.replace('\\','\\')
    # plot OLCI
    S3plots.olci_scatter(xol, yol, chl_oc4me, shppath, bound, 'CHL_OC4ME ' + fdate, plotpath)
    
    del chl_oc4me, xol, yol
