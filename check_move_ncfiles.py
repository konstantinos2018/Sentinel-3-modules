# -*- coding: utf-8 -*-

# =============================================================================
# Check SLSTR
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
#import matplotlib.pyplot as plt
import os
import datetime as dt
import pdb
import numpy as np
# My Modules
from s3utilities import mv_folders_files
from S3postproc import check_npempty
import nc_manipul as ncman
import S3postproc
import S3coordtran as s3ct

# File directory
path = r'D:\vlachos\DOCUME~1\KVMSCT~1\Data\SATELL~1\GULFST~1\SLSTR\SLSTR_~1'.replace('\\','\\') # Gulf Stream 1
folder = os.listdir(path)

# Plot and check which data are useful
lst = ['sea_surface_temperature', 'l2p_flags', 'quality_level']

# Coordinate Reference Systems definition
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1

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
    
    if np.all(np.isnan(sst)) == True or check_npempty(sst):
        # Add bad filename to list
        no_data.append(f)
        continue
    
# Give source and destination directories
dir_src = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR\SLSTR_extra'.replace('//','//')
dir_dst = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR\SLSTR_extra\out'.replace('//','//')

# Move bad files to folder
mv_folders_files(dir_src, dir_dst, no_data)

#%%
# =============================================================================
# Check OLCI
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import numpy as np
import pdb
# My Modules
from s3utilities import mv_folders_files
import nc_manipul as ncman
import S3postproc
import S3coordtran as s3ct

#path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\') # North Sea
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI\OLCI_extra'.replace('\\','\\') # Gulf Stream

folder = os.listdir(path)
fname = 'sub_OLCI.nc'

# Plot and check which data are useful
lst = ['CHL_OC4ME', 'WQSF']

# Coordinate Reference Systems definition
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

bad = []
no_data = []

for f in folder:
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
    
    if np.all(np.isnan(chl_oc4me)) == True:
        no_data.append(f)
        continue

# Give source and destination directories
dir_src = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI\OLCI_extra'.replace('//','//')
dir_dst = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI\OLCI_extra\out'.replace('//','//')

# Move bad files to folder
mv_folders_files(dir_src, dir_dst, no_data)

#%%
# =============================================================================
# Check SRAL
# =============================================================================
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import numpy as np
import pdb
# My Modules
import nc_manipul as ncman
import S3postproc
import S3coordtran as s3ct
from s3utilities import mv_folders_files

#'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\') # North Sea
#path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\SRAL'.replace('\\','\\') # Mediterranean
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL\SRAL_extra'.replace('\\','\\') # Gulf Stream 1
folder = os.listdir(path)
fname = 'sub_enhanced_measurement.nc'

# Plot and check which data are useful
lst = ['ssha_20_ku', 'flags']

# Coordinate Reference Systems definition
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

#bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

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
    
    if np.all(np.isnan(ssha_m)) == True:
        no_data.append(f)
        continue

# Give source and destination directories
dir_src = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL\SRAL_extra'.replace('//','//')
dir_dst = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL\SRAL_extra\out'.replace('//','//')

# Move bad files to folder
mv_folders_files(dir_src, dir_dst, no_data)
    