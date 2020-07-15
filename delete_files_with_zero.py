# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:48:53 2018

@author: vlachos
"""
# =============================================================================
# DESCRIPTION
# The purpose of this script is to delete all the NetCDF files that their variables
# have zero size. The variables that are used as flag-variables are ssha_20_ku
# for SRAL, chl_nn for OLCI and sea_surface_temperature for SLSTR
# Each block of the script needs to be run separately
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
from netCDF4 import Dataset
import os
import shutil
#import nc_manipul as ncman
#import S3postproc
#import S3plots
#import S3coordtran as s3ct
#import datetime as dt
#import matplotlib.pyplot as plt

path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL\SRAL_extra'.replace('\\','\\')
folder = os.listdir(path)
fname = 'sub_enhanced_measurement.nc'

for f in folder:
    fullpath = os.path.join(os.path.join(path, f), fname)
    nc = Dataset(fullpath)
    var = nc.variables['ssha_20_ku'][:]
    
    if var.size == 0:
        nc.close()
        print(f)
        shutil.rmtree(os.path.join(path, f))
    else:
        nc.close()
    
#%%
# =============================================================================
# IMPORTS
# =============================================================================
#import os
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
import os
import shutil
#import nc_manipul as ncman
#import S3postproc
#import S3plots
#import S3coordtran as s3ct
#import datetime as dt

path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI\OLCI_extra'.replace('\\', '\\')
folder = os.listdir(path)
fname = 'sub_OLCI.nc'

for f in folder:
    fullpath = os.path.join(os.path.join(path, f), fname)
    nc = Dataset(fullpath)
    var = nc.variables['CHL_NN'][:]
    
    if var.size == 0:
        nc.close()
        print(f)
        shutil.rmtree(os.path.join(path, f))
    else:
        nc.close()

#%%
# =============================================================================
# IMPORTS
# =============================================================================
from netCDF4 import Dataset
#import matplotlib.pyplot as plt
import os
import shutil
#import nc_manipul as ncman
#import S3postproc
#import S3plots
#import S3coordtran as s3ct
#import datetime as dt

path = r'D:\vlachos\DOCUME~1\KVMSCT~1\Data\SATELL~1\GULFST~1\SLSTR\SLSTR_~1'.replace('\\','\\')
folder = os.listdir(path)

for f in folder:
    fname = os.listdir(os.path.join(path, f))[0]
    fullpath = os.path.join(os.path.join(path, f), fname)
    nc = Dataset(fullpath)
    var = nc.variables['sea_surface_temperature'][:]
    
    if var.size == 0:
        nc.close()
        print(f)
        shutil.rmtree(fullpath)
    else:
        nc.close()
    
