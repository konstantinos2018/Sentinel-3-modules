# -*- coding: utf-8 -*-
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
from netCDF4 import Dataset
import datetime as dt
import os
import numpy as np
from astropy.convolution import convolve as astro_conv
from S3postproc import check_npempty
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import geopandas as gpd
from matplotlib.collections import LineCollection
import time
import pdb
# My Modules
import S3coordtran as s3ct
import nc_manipul as ncman
import S3postproc


sst_level4_path = r'D:\vlachos\Documents\KV MSc thesis\Data\SST_Level_4\aggregate__ghrsst_JPL_OUROCEAN-L4UHfnd-GLOB-G1SST_OI.ncml.nc'.replace('\\','\\')

sral_path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\')
sral_files = os.listdir(sral_path)

# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream 1

bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream 1

shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1

polys = gpd.read_file(shppath)
# Define colors for shapefile
cbrown = '#CD853F'
cblack = '#000000'
        
# Open netCDF
nc = Dataset(sst_level4_path)

lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij') # create meshgrid
rows, cols = lon_grid.shape
lon_grid = np.matrix.flatten(lon_grid) # convert to array
lat_grid = np.matrix.flatten(lat_grid) # convert to array
# transform coordinates
x_sst, y_sst = s3ct.sral_coordtran(lon_grid, lat_grid, inEPSG, outEPSG)
x_sst = np.reshape(x_sst, (rows, cols))
y_sst = np.reshape(y_sst, (rows, cols))
del lon, lat, lon_grid, lat_grid

#sst_tseries = nc.variables['analysed_sst'][:] - 273.15
#time = nc.variables['time'][:] # seconds since 1981-01-01 00:00:00

fname_sral = 'sub_enhanced_measurement.nc'
lst_sral = ['ssha_20_ku', 'flags']
bad_sral = []

total_iteration = nc.dimensions['time'].size

font = {'size' : 18}
plt.rc('font', **font)

# Plot common dates
for i in range(nc.variables['time'].shape[0]):
    plt.close('all')
    print('File: {0} out of {1}\n'.format(i, total_iteration))
    if i < 291:
        continue
    
    t_start = time.time()
    
    f_sst_time = nc.variables['time'][i] # seconds since 1981-01-01 00:00:00
    f_sst_time = dt.datetime.fromtimestamp(f_sst_time + dt.datetime(1981,1,1).timestamp()-dt.timedelta(hours=1).total_seconds())
#        if (f_sst_time < dt.datetime(2019, 1, 26)) or (f_sral[:3] == 'S3A'):
#            continue
    f_sst_time_check = f_sst_time.strftime('%Y%m%d')
    
    # -------- SST Level 4
    sst_level4 = nc.variables['analysed_sst'][i] - 273.15
    sst_level4_mask = nc.variables['mask'][i] == 1 # Read Sea mask
    sst_level4[~sst_level4_mask] = np.nan # Apply mask
    
    f_sst_time = dt.datetime.strptime(f_sst_time_check, '%Y%m%d')
    f_sst_time = f_sst_time.strftime('%Y-%m-%d')
            
    # Create figure and take axis handle
    fig = plt.figure(figsize=(18, 10))
    ax = fig.gca()
    
    polys = polys.geometry

    for poly in polys :
        poly_patch = PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=1,zorder=0)
        ax.add_patch(poly_patch)
    
    cm_sstL4 = plt.cm.get_cmap('RdYlBu_r')
    sc = ax.pcolor(x_sst,y_sst, sst_level4, cmap=cm_sstL4)#, vmin=vmin, vmax=vmax)#, norm=norm)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel('SST [$^\circ$C]', rotation=270, labelpad=20, fontsize=18)
    ax.set_xlim(-3000000, -1000000)
    ax.set_ylim(3625000, 4875000)
    ax.set_xlabel('X [m]', fontsize=18)
    ax.set_ylabel('Y [m]', fontsize=18)
    plt.xticks(rotation=45)
            
    del sst_level4_mask

    print('SST LEVEL4: OK')
    
    dates_sral = []        
    for f_sral in sral_files:
        if (f_sst_time_check == f_sral[16:24]):
            print('1')
#            if dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S') != dt.datetime(2017, 12, 16, 20, 00, 57):                
#                break
#                if f_sral[:3] != 'S3B':
#                    break
            
            # ========== SRAL                    
            fullpath = os.path.join(os.path.join(sral_path, f_sral), fname_sral)
            # Read netcdf
            try:
                lonsr, latsr, ssha, flagsr = ncman.sral_read_nc(fullpath, lst_sral)
            except:
                # Keep name of file which was not read correctly
                bad_sral.append(f_sral)
                continue
            
            # transform coordinates
            xsr, ysr = s3ct.sral_coordtran(lonsr, latsr, inEPSG, outEPSG)
            del lonsr, latsr
            # subset dataset
            xsr, ysr, ssha, flagsr = ncman.sral_subset_nc(xsr, ysr, ssha, flagsr, bound)
            ssha = ssha['ssha_20_ku']
            # Apply flags/masks
            _, outmask_ssha = S3postproc.apply_masks_sral(ssha, 'ssha_20_ku', flagsr)
            
            # Clear
            del flagsr
            # Check if empty
            if check_npempty(ssha):
                print('SSHA | date {0} is empty'.format(f_sst_time))
                continue
            
            # =============================================================================
            # SSHA OUTLIER DETECTION            
            # =============================================================================
            # Choose inside percentiles
#            Q1, Q3 = np.nanpercentile(ssha,q=[25,75], interpolation='linear')
#            IQR = Q3 - Q1 # Interquartile range
            thresh_low = np.nanpercentile(ssha,q=[10], interpolation='linear')
            thresh_up = np.nanpercentile(ssha,q=[90], interpolation='linear')
            
            # Outlier mask
            idx = (ssha > thresh_low) & (ssha < thresh_up)
            # Outlier and flag mask
            idx = idx & outmask_ssha
            
            # =============================================================================
            # FILTER SSHA
            # =============================================================================
            log_window_size = []
            ssha[~idx] = np.nan
            window_size = [35]
            for ws in window_size:
                # Check window size
                if ssha.size < ws:
                    ws = ssha.size
                    # Check if window size is odd or even (needs to be odd)
                    if ws % 2 == 0:
                        ws = ws + 1
                    # Log which files do not use the default window size
                    log_window_size.append(f_sral)
                else:
                    pass
                ssha_m = astro_conv(ssha, np.ones((ws))/float(ws), boundary='extend',
                                    nan_treatment='interpolate', preserve_nan=False)
            
            ssha_m = S3postproc.rescale_between(ssha_m, -1, 1)
            
            print('SSHA: OK')
            
            # SRAL
            # segments
            points = np.array([xsr, ysr]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='jet', norm=norm)
            
            lc.set_alpha(1)
            lc.set_array(ssha_m)
            lc.set_linewidth(8)
            line = ax.add_collection(lc)
            
            fdate_sral = dt.datetime.strptime(f_sral[16:31], '%Y%m%dT%H%M%S')
            fdate_sral = fdate_sral.strftime('%Y-%m-%d %H_%M_%S')            


            
            # Keep SRAL dates that are used
            dates_sral.append(f_sral[:3] + '_' + fdate_sral)
            
            del ssha_m, ssha
    if 'line' in locals():
        pass
    else:
        continue
    cbar = plt.colorbar(line, ax=ax)
    cbar.ax.set_ylabel('SSHA [m]', rotation=270, labelpad=10, fontsize=18)
    del sst_level4
    
    # filename
    filename = '{0}_SST_L4'.format(f_sst_time)
    plotpath = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\SRAL_SST_LEVEL4'.replace('\\', '\\')
    plot_title = 'SST_L4_{0}\n{1}'.format(f_sst_time, '\n'.join(dates_sral))    
    plt.title(plot_title, fontsize=23)
    plt.savefig(plotpath + '\\' + filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    t_end = time.time()
    print('Time take for this run is: {0:.2f} s'.format(t_end - t_start))
    
    del sc
    
            