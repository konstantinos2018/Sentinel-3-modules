# -*- coding: utf-8 -*-

# =============================================================================
# SLSTR
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import datetime as dt
import pdb
import numpy as np
# My Modules
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct

# File directory
#path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\SLSTR'.replace('\\','\\') # Mediterranean
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\') # Gulf Stream 1
folder = os.listdir(path)

# Plot and check which data are useful
lst = ['sea_surface_temperature', 'l2p_flags', 'quality_level']

inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ' # epsg: 4326
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ' # Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream

bad_1 = []
bad_2 = []
no_data = []
for f in folder:
    try:
        fname = os.listdir(os.path.join(path, f))
        fullpath = os.path.join(os.path.join(path, f), fname[0])
    except:
        bad_1.append(f)
        continue
    # Read netcdf
    try:
        lonsl, latsl, varValues, l2p_flags, quality_level = ncman.slstr1D_read_nc(fullpath, lst)
    except:
        bad_2.append(f)
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
    sst, outmasks = S3postproc.apply_masks_slstr(sst, 'sea_surface_temperature', masks, quality_level)
    del masks
    
    # Apply flag masks
    xsl = xsl[outmasks]
    ysl = ysl[outmasks]
    # Apply varValues (e.g. sst) masks
    xsl = xsl[sst.mask]
    ysl = ysl[sst.mask]
    sst = sst.data[sst.mask] - 273 # convert to Celsius
    
    fdate = dt.datetime.strptime(f[16:31], '%Y%m%dT%H%M%S')
    fdate = fdate.strftime('%Y-%m-%d %H_%M_%S')
    
    if np.all(np.isnan(sst)) == True:
#        pdb.set_trace()
        no_data.append(f)
        continue
    
    plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SLSTR'.replace('\\','\\')
    # plot SLSTR
    S3plots.slstr_scatter(xsl, ysl, sst, shppath, bound, f[:3] + '_' + fdate + '_SST', plotpath)

    del sst, xsl, ysl, outmasks, varValues
