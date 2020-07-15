# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:17:25 2019

@author: vlachos
"""
# =============================================================================
# DESCRIPTION
# The following script makes a comparison between different 2D moving averaging
# filtering windows using the manual approach for the SLSTR.
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import nc_manipul as ncman
import os
from scipy import spatial
import datetime as dt
import pdb
# My modules
import S3coordtran as s3ct
import S3postproc as s3postp
import S3plots as s3plot

# ========== Initialize general use variable ========
# General definitions
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

## Define colors for shapefile
#cbrown = '#CD853F'
#cblack = '#000000'

# Geographical boundaries
bound = [3500000, 4300000, 3100000, 4000000] # North Sea
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream

# Read shapefile
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\Mediterannean_23031.shp'.replace('\\', '\\') # Mediterranean_TEST

# File directory
slpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\') # North Sea
#slpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Mediterranean_test\SLSTR'.replace('\\','\\') # Mediterranean
fold_sl = r'S3A_SL_2_WST____20180507T201349_20180507T215448_20180509T052750_6059_031_042______MAR_O_NT_003.SEN3' # North Sea
#fold_sl = r'S3A_SL_2_WST____20180714T205109_20180714T223209_20180716T060524_6059_033_242______MAR_O_NT_002.SEN3' # Mediterranean
fname_sl = os.listdir(os.path.join(slpath, fold_sl))[0]
path = os.path.join(os.path.join(slpath, fold_sl), fname_sl)

var = ['sea_surface_temperature', 'l2p_flags', 'quality_level']

# Read netcdf
lon, lat, varValues, l2p_flags, quality_level = ncman.slstr1D_read_nc(path, var)
# transform coordinates
xsl, ysl = s3ct.slstr_olci_coordtran(lon, lat, inEPSG, outEPSG)

# subset dataset
varValues = ncman.slstr_olci_subset_nc(xsl, ysl, varValues, bound)
# Extract bits of the l2p_flags flag
flag_out = s3postp.extract_bits(l2p_flags, 16)
# Extract dictionary with flag meanings and values
l2p_flags_mean, quality_level_mean = s3postp.extract_maskmeanings(path)
# Create masks
masks = s3postp.extract_mask(l2p_flags_mean, flag_out, 16)
# Apply masks to given variables
# Define variables separately
sst = varValues['sea_surface_temperature']
sst, outmasks = s3postp.apply_masks_slstr(sst, 'sea_surface_temperature', masks, quality_level)

# Apply flag masks
xsl = xsl[outmasks]
ysl = ysl[outmasks]
# Apply varValues (e.g. sst) masks
xsl = xsl[sst.mask]
ysl = ysl[sst.mask]
sst = sst.data[sst.mask] - 273 # convert to Celsius

# FILTER 2D
XY_query = np.dstack([xsl, ysl])[0]
XY_query = np.array_split(XY_query, 500) # Segment
XY_obs = np.dstack([xsl, ysl])[0]
tree_obs = spatial.cKDTree(XY_obs) # tree of observations (sst)

sst_movAv = [] # Initialization
i=0
radi = 150000 # meters
for segment in XY_query:
    i = i + 1
    print(i)
    tree_query = spatial.cKDTree(segment) # tree of query points
    out = tree_query.query_ball_tree(tree_obs, r=radi)

    for point in out:
        sst_movAv.append(np.nanmean(sst[point]))
#    pdb.set_trace()

sst_movAv = np.asarray(sst_movAv)
fdate = dt.datetime.strptime(fold_sl[16:31], '%Y%m%dT%H%M%S')
fdate = fdate.strftime('%Y-%m-%d %H_%M_%S')
    
# plot SLSTR
s3plot.slstr_scatter(xsl, ysl, sst_movAv, shppath, bound, 'SST_after_'+str(radi/1000.0)+'km '+fdate)
