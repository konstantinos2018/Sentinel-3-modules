# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:04:27 2019

@author: vlachos
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:25:33 2019

@author: vlachos
"""

# =============================================================================
# DESCRIPTION:
# The following script makes a comparison between different moving averaging
# filtering windows using the ASTROPY package. More options can be explored
# such as interpolation of NaNs, among others, for the SRAL
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
import os
import nc_manipul as ncman
import S3postproc
import S3plots
import S3coordtran as s3ct
import datetime as dt
import numpy as np
import pdb
import scipy.stats as scst
import scipy.signal as scsign
import matplotlib.pyplot as plt
from astropy.convolution import convolve as astro_conv
from astropy.convolution import interpolate_replace_nans, Gaussian1DKernel

#path= r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\') # North Sea
path = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\') # Gulf Stream
folder = os.listdir(path)
fname = 'sub_enhanced_measurement.nc'

# Plot and check which data are useful
lst = ['ssha_20_ku', 'flags']

# Coordinate Reference Systems definition
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea

# Bounding box region coordinates
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1
#bound = [3500000, 4300000, 3100000, 4000000] # North Sea

# country shapefile directory
#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\North_Sea_BorderCountries_3035.shp'.replace('\\', '\\') # North Sea
#shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\Mediterannean_23031.shp'.replace('\\', '\\') # Mediterranean_TEST

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
    del flagsr#, ssha
 
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
#    # Compute percentiles
    idx = (ssha_m > np.percentile(ssha_m, 5)) & (ssha_m < np.percentile(ssha_m, 95))
#    ssha_m = ssha_m[idx]

    
    # Compute distances between SRAL points
    dst = S3postproc.sral_dist(xsr, ysr)
    
    # Initialize dictionary with variables
    var = {}
    legend_labels = []
    # =============================================================================
    #     FILTERS
    # =============================================================================
    # Original (if used) must be the 1st element
    
    window_size = [11, 21, 35, 101, 201, 333, 455] # define windows sizes you want
    window_size.sort() # sort in increasing order
    labels = ['MovAv_'+str(item*330.0/1000)+' km' for item in window_size] # Create list of strings with names
    
    line_props = [
            {'color':'y', 'linestyle':'-', 'alpha':1, 'linewidth':1.3},
            {'color':'g', 'linestyle':'-', 'alpha':0.6, 'linewidth':1.3},
            {'color':'r', 'linestyle':'-', 'alpha':0.6, 'linewidth':2},
            {'color': '#ffcdc3', 'linestyle': '-', 'alpha':0.8, 'linewidth':3},
            {'color':'k', 'linestyle':'-', 'alpha':0.6, 'linewidth':3},
            {'color':'m', 'linestyle':'-', 'alpha':0.6, 'linewidth':3},
            {'color':'c', 'linestyle':'-', 'alpha':0.6, 'linewidth':4}
            ]
    
    # Fix nans
#    kernel = Gaussian1DKernel(2)
#    ssha_fixed = interpolate_replace_nans(ssha_m, kernel)
    ssha_m[~idx] = np.nan
    
    if True:
        window_size.insert(0, 0) # insert 0 for Original ssha
        labels.insert(0, 'Original') # Insert in the 0th element
        line_props.insert(0, {'color':'b', 'linestyle':'-', 'alpha':0.6, 'linewidth':0.8})
    
    for wind, lab, prop in zip(window_size, labels, line_props):
        # Do for original
        if lab == 'Original':
            var[lab] = [ssha_m,
               prop]
            legend_labels.append(lab)
        elif ssha_m.size >= wind:
            var[lab] = [astro_conv(ssha_m, np.ones((wind))/float(wind), boundary='extend',
               nan_treatment='interpolate',preserve_nan=True),
               prop]
            legend_labels.append(lab)


#    ssha[outmask | idx] = np.nan
#    dst = S3postproc.sral_dist(xsr, ysr)    
    # Insert nans
#    dst_sr_nan, ssha_m_nan, idx_nan = S3postproc.sral_dist_nans(dst_sr, ssha_m, threshold=3*340)
    
    legend_labels = tuple(legend_labels)
    
    plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\SRAL_clean\Trajectories\Filter_comparison'.replace('\\','\\')
    # Plot SRAL
    S3plots.multiple_cross_sections(var, dst, legend_labels, fdate, plotpath)
    