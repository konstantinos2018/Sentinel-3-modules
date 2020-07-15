# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 12:32:39 2018

@author: vlachos
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import shapefile as shp
from descartes import PolygonPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pdb
import matplotlib.colors as colors
# My Modules
from S3coordtran import sral_coordtran
from nc_manipul import sral_read_nc

path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Test data\S3-A\S3A_OLCI\S3A_OL_2_WFR____20181101T091611_20181101T091911_20181102T174142_0179_037_264_2340_MAR_O_NT_002.SEN3'.replace('\\','\\')
fname = 'wqsf.nc'
varName = 'WQSF'
nc = Dataset(os.path.join(path, fname))

wqsf = nc.variables[varName][:]

nc = Dataset(os.path.join(path, 'geo_coordinates.nc'))
lat2 = nc.variables['latitude'][:]
lon2 =  nc.variables['longitude'][:]

nc = Dataset(os.path.join(path, 'chl_nn.nc'))
chl_nn2 =  nc.variables['CHL_NN'][:]

#%%
# =============================================================================
# DESCRIPTION
# Plot altimetry ground tracks based on given date ranges
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

# General modules
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pdb
import matplotlib.colors as colors
import scipy.signal as scsign
from astropy.convolution import convolve as astro_conv
import datetime as dt
import gepandas as gpd
from descartes import PolygonPatch
# My Modules
import S3coordtran as s3ct
import S3postproc as s3postp
import S3plots as s3plot
import s3utilities
import nc_manipul as ncman

# ========== Initialize general use variable ========
# General definitions
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
#outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream

# Define colors for shapefile
cbrown = '#CD853F'
cblack = '#000000'

# Bounding box region coordinates
bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream
bound = [3500000, 4300000, 3100000, 4000000] # North Sea

# Read shapefile
shppath = r"D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp".replace('\\','\\')
polys = gpd.read_file(shppath)
polys = polys.geometry[0]

# =================== Read SRAL =====================
# File directory
srpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\')
#srpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\')
srfileName = 'sub_enhanced_measurement.nc'
# Folder in srpath
folders = os.listdir(srpath)

# ==== Create start times
# Create time step
t_step = dt.timedelta(26) # number of days
t_threshold = dt.datetime(2018, 11, 30)
# Start date
t_start = [dt.datetime(2017, 11, 01)]
item = t_start[0] + t_step
while item <= t_threshold:
    t_start.append(item)
    item = item + t_step
# ===== Create end times
# End date
t_end = [i+t_step-dt.timedelta(1) for i in t_start]

# Convert times to strings
t_start = [i.strftime('%Y%m%d') for i in t_start]
t_end = [i.strftime('%Y%m%d') for i in t_end]

counter = 0
#lowcount = 2
#upcount = 3


# Track direction ('ascending', 'descending', 'both')
track_dir = 'both'

lst = ['ssha_20_ku', 'flags']
ddd = []
for st, en in zip(t_start, t_end):
    # Create figure and take axis handle
    fig = plt.figure(figsize=(18,10))
    ax = fig.gca()
    # colorbar colormap
    cm = plt.cm.get_cmap('cool_r')
    
    # Plot shapefile
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    
    for fold in folders:  
        if fold[:3] == 'S3A':
            pass
        else:
            continue
        try:
            if (fold[16:24] >= st) & (fold[16:24]<= en):#dirs[16:24] < 40:
                counter = counter + 1
                
                # XML file parsing (decide ascending, descending or both)
                condition = s3utilities.read_xml_S3(os.path.join(srpath, os.path.join(fold, 'xfdumanifest.xml')), track_direction=track_dir)
                if condition==True or condition==False:
                    
                    # Read netcdf
                    lon, lat, ssha, flagsr = ncman.sral_read_nc(os.path.join(srpath, os.path.join(fold, srfileName)), lst)
                    
                    # transform coordinates
                    xsr, ysr = s3ct.sral_coordtran(lon, lat, inEPSG, outEPSG)
        #            ssha = ssha['ssha_20_ku']
                    
                    xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)
                    # Apply flags/masks
                    ssha_m, outmask = s3postp.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
                    # Clear
                    del flagsr
                    
                    # Apply outmask
                    xsr = xsr[outmask]
                    ysr =  ysr[outmask]
                    del outmask
                    
                    # Compute distance
                    dst_sr = s3postp.sral_dist(xsr, ysr)
#                    print dst_sr.size
                    ddd.append(dst_sr.max())
                    
                    # Filter ssha
                    # ====== Astropy Moving Average
                    # Choose inside percentiles
                    idx = (ssha_m > np.percentile(ssha_m, 20)) & (ssha_m < np.percentile(ssha_m, 80))
                    log_window_size = []
                    ssha_m[~idx] = np.nan
                    window_size = 35
                    # Check window size
                    if ssha_m.size < window_size:
                        window_size = ssha_m.size
                        # Check if window size is odd or even (needs to be odd)
                        if window_size % 2 == 0:
                            window_size = window_size + 1
                        # Log which files do not use the default window size
                        log_window_size.append(fold)
                        
                    ssha_m_filt = astro_conv(ssha_m, np.ones((window_size))/float(window_size), boundary='extend',
                                        nan_treatment='interpolate',preserve_nan=False)
                    # Rescale
                    ssha_m_filt = s3postp.rescale_between(ssha_m_filt, -1, 1)
                    
                #            # Plot variable
                #            sc = plt.scatter(xsr, ysr, c=ssha_m, marker='.', s=8**2, cmap=cm,
                #                             vmin=-0.2, vmax=0.5)
#                    # segments
#                    points = np.array([xsr, ysr]).T.reshape(-1, 1, 2)
#                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
#                    
#                    norm = plt.Normalize(-1, 1)#0.2, 0.5)
#                    lc = LineCollection(segments, cmap='jet', norm=norm)
#                    
#                    lc.set_alpha(1)
#                    lc.set_array(ssha_m_filt)
#                    lc.set_linewidth(8)
#                    line = ax.add_collection(lc)

        except Exception as e:
            print str(e)
            continue
#    ddd = np.asarray(ddd)
#    cbar = plt.colorbar(line, ax=ax)
#    cbar.ax.set_ylabel('SSHA [m]', rotation=270, fontsize=18)#, labelpad=-10)
#    #cbar = plt.colorbar(sc)#, cax=cbaxes)
#    #cbar.ax.set_ylabel('SSHA [m]', rotation=270)
#    
#    # Tick labels size
#    font = {'size' : 18}
#    plt.rc('font', **font)
#        
#    plt.title('SRAL '+'-'.join([st[0:4], st[4:6], st[6:]])+' to '+'-'.join([en[0:4], en[4:6], en[6:]]), fontsize=23)
#    
#    # Set axis limits
#    plt.xlim(bound[0],bound[1])
#    plt.ylim(bound[2],bound[3])
#    
#    # Labels
#    plt.xlabel('X [m]', fontsize=18)
#    plt.ylabel('Y [m]', fontsize=18)
#    
#    # Rotate axis labels
#    plt.xticks(rotation=45)
#    
#    # Save plot
#    outpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\North_Sea\SRAL_clean\Multitemporal'.replace('\\','\\')
#    plt.savefig(outpath+'\\'+track_dir+' '+st+'_'+en, dpi=300)
#    
#    plt.close('all')



