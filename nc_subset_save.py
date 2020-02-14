# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 21:12:13 2018

@author: vlachos
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import sys
import re
#import re
import netCDF4 as netc
import numpy as np
import numpy.matlib
import traceback
import pdb

# =============================================================================
# BEGIN
# =============================================================================
def ncSubset_save(src, dst, bound, sensor, rmv=False):
    ## Lat lon to variables
    #lat = nc.variables['lat_20_ku'][:]
    #lon = nc.variables['lon_20_ku'][:]
    
    counter = 0
    n = len(os.listdir(src))
    
    if sensor == 'SLSTR':
        # subset all files' variables in folder
        for subdir, dirs, files in os.walk(src):
            # progress
            sys.stdout.write("\rProgress...{0:.2f}%".format((float(counter)/n)*100))
            sys.stdout.flush()
                    
            for f in files:
                if '.nc' in f:
                    
                    # Read latitude and longitude
                    nc = netc.Dataset(os.path.join(subdir, f))
                    lat = nc.variables['lat'][:]
                    lon = nc.variables['lon'][:]
                    nc.close()
                    # Create mask
                    mask = (lon > bound[2]) & (lon < bound[3]) & (lat > bound[0]) & (lat < bound[1])
                    
        #            print(os.path.join(subdir,f))
                    # Read nc file and create the new nc file
                    orig = netc.Dataset(os.path.join(subdir, f))
                    copy = netc.Dataset(subdir+'\\sub_'+f, 'w')
        #            print(orig)
                    # Copy the attribute names
                    for attr in orig.ncattrs():
                        copy.setncattr(attr, orig.getncattr(attr))
                    
                    # Initialize empty dictionary
                    copyVars = {}
          
                    for var in orig.variables:
                        # read variable metadata, check shape and copy to new netCDF
                        if len(orig.variables[var].shape) == 1:
                            copyVars[var] = orig.variables[var][:]
                            
                        elif len(orig.variables[var].shape) == 2:
                            copyVars[var] = np.extract(mask[:], orig.variables[var][:])
                            
                        elif len(orig.variables[var].shape) > 2:
        #                    print(var)
                            copyVars[var] = np.extract(mask[:], orig.variables[var][:][-2:])
                            
          
                    for var in copyVars:
        #                print('here')
                        copy.createDimension('dim'+var, copyVars[var].size)
        #                print('there')
                        v = copy.createVariable(var, orig.variables[var].datatype, 'dim'+var)
            
                        for attr in orig.variables[var].ncattrs():
                            v.setncattr(attr, orig.variables[var].getncattr(attr))
            
                        v[:] = copyVars[var][:]
                        
                    orig.close()
                    copy.close()
                    if rmv:
                        # delete original
                        os.remove(os.path.join(subdir, f))
                    
                    counter = counter + 1
    elif sensor == 'SRAL':
        # subset all files' variables in folder
        for subdir, dirs, files in os.walk(src):
            # progress
            sys.stdout.write("\rProgress...{0:.2f}%".format((float(counter)/n)*100))
            sys.stdout.flush()
                    
            for f in files:
                if '.nc' in f:
                    
                    # Read latitude and longitude
                    nc = netc.Dataset(os.path.join(subdir, f))
                    
                    # lat, lon 1Hz
                    lat01 = nc.variables['lat_01'][:]
                    lon01 = nc.variables['lon_01'][:]
                    # lat, lon 20Hz C-band
                    lat20c = nc.variables['lat_20_c'][:]
                    lon20c = nc.variables['lon_20_c'][:]
                    # lat, lon 20Hz Ku-band
                    lat20ku = nc.variables['lat_20_ku'][:]
                    lon20ku = nc.variables['lon_20_ku'][:]
                    
                    # Correct if the coordinates are greater than 180
                    if np.any(lon01 > 180):
                        # lon 1Hz
                        lon01 = lon01 - 360
                        # lon 20Hz C-band
                        lon20c = lon20c - 360
                        # lon 20Hz Ku-band
                        lon20ku = lon20ku - 360
                    
                    nc.close()
                    
                    # Create mask
                    mask01 = (lon01 > bound[2]) & (lon01 < bound[3]) & (lat01 > bound[0]) & (lat01 < bound[1])
                    mask20c = (lon20c > bound[2]) & (lon20c < bound[3]) & (lat20c > bound[0]) & (lat20c < bound[1])
                    mask20ku = (lon20ku > bound[2]) & (lon20ku < bound[3]) & (lat20ku > bound[0]) & (lat20ku < bound[1])
                    
        #            print(os.path.join(subdir,f))
                    # Read nc file and create the new nc file
                    orig = netc.Dataset(os.path.join(subdir, f))
                    copy = netc.Dataset(subdir+'\\sub_'+f, 'w')
        #            print(orig)
                    # Copy the attribute names and values
                    for attr in orig.ncattrs():
                        copy.setncattr(attr, orig.getncattr(attr))
                    
                    # Initialize empty dictionary
                    copyVars = {}
          
                    for var in orig.variables:
                        # read variable metadata, check shape and copy to new netCDF
                        if orig.variables[var].size == len(lat01):
                            copyVars[var] = np.extract(mask01, orig.variables[var][:])
                            
                        elif orig.variables[var].size == len(lat20c):
                            copyVars[var] = np.extract(mask20c, orig.variables[var][:])
                            
                        elif orig.variables[var].size == len(lat20ku):
                            copyVars[var] = np.extract(mask20ku, orig.variables[var][:])
                            
                        else:
                            # Put the original values if the shape is other than 1D
                            copyVars[var] = orig.variables[var][:]
                            
          
                    for var in copyVars:
        #                print('here')
                        copy.createDimension('dim'+var, copyVars[var].size)
        #                print('there')
                        v = copy.createVariable(var, orig.variables[var].datatype, 'dim'+var)
            
                        for attr in orig.variables[var].ncattrs():
                            v.setncattr(attr, orig.variables[var].getncattr(attr))
            
                        v[:] = copyVars[var][:]
                        
                    orig.close()
                    copy.close()
                    
                    # delete original
                    if rmv:
                        os.remove(os.path.join(subdir, f))
                    
                    counter = counter + 1
    elif sensor == 'OLCI':
        
        # define list with kick-out netCDFs
        outvar = ['tie_geo_coordinates.nc', 'tie_geometries.nc',
                  'tie_meteo.nc', 'xfdumanifest.xml', 'instrument_data.nc']
        counter = 0
        
        # Read time and make it a 2d array based on lon-lat
        # subset all files' variables in folder
        for subdir, dirs, files in os.walk(src):
            try:
                # progress
                sys.stdout.write("\rProgress...{0:.2f}%".format((float(counter)/n)*100))
                sys.stdout.flush()

                # Read coordinates and create mask
                nc = netc.Dataset(os.path.join(subdir, 'geo_coordinates.nc'))
#                return nc
#                break
                # lat, lon
                lat = nc.variables['latitude'][:]
                lon = nc.variables['longitude'][:]
                nc.close()
                
                # create mask
                mask = (lon > bound[2]) & (lon < bound[3]) & (lat > bound[0]) & (lat < bound[1])
                
                # Read time and make 2darray
                nc = netc.Dataset(os.path.join(subdir, 'time_coordinates.nc'))
                time = nc.variables['time_stamp'][:]
                # Reshape to column vector
                time = np.reshape(time, (time.shape[0], 1))
                time = numpy.matlib.repmat(time, 1, lat.shape[1])
                
                
                # kick-out unuseful netcdfs
                files = list(set(files) - set(outvar))
                
                # Create new netcdf file
                copy = netc.Dataset(os.path.join(subdir,'sub_OLCI.nc'), 'w')
                
                # Copy the attribute names and values
                for attr in nc.ncattrs():
                    if attr == 'title':
                        s = nc.getncattr(attr).replace(',', '')
                        s = s.split(' ')
                        s = ' '.join(s[0:5])
                        copy.setncattr(attr, s)

                    else:
                        copy.setncattr(attr, nc.getncattr(attr))
                        
                # Close time netcdf file
                nc.close()
                
                # Run in every netcdf and subet it
                for f in files:
                    
                    # open original netcdf
                    orig = netc.Dataset(os.path.join(subdir, f))
                    
                    # Initialize empty dictionary which will contain the variables
                    copyVars = {}
                    
                    # read variable metadata, check shape and copy to new netCDF
                    # Pass the masked variables into the copyVars dictionary
                    try:
                        if f == 'time_coordinates.nc':
                            for var in orig.variables:
                                copyVars[var] = np.extract(mask, time)
                        else:
                            for var in orig.variables:
                                copyVars[var] = np.extract(mask, orig.variables[var][:])
                    except:
                        pdb.set_trace()
                            
                    for var in copyVars:
                        # Create dimension of each of the copy variables
                        copy.createDimension('dim'+var, size=copyVars[var].size)
                        # Create the variables of the copy file
                        v = copy.createVariable(var, datatype=orig.variables[var].datatype, dimensions='dim'+var, 
                                                chunksizes=(copyVars[var].size/25,),
                                                zlib=orig.variables[var].filters()['zlib'],
                                                complevel=orig.variables[var].filters()['complevel'])
                        # copy the variables attributes from original to copy
                        for attr in orig.variables[var].ncattrs():
                            v.setncattr(attr, orig.variables[var].getncattr(attr))
                        
                        v[:] = copyVars[var][:]
                    # Close original netcdf
                    orig.close()
                    
                    # delete original
                    if rmv:
                        os.remove(os.path.join(subdir, f))
                
                # Increase counter
                counter = counter + 1
                
                # Close copy netcdf
                copy.close()
                
            except Exception:
                traceback.print_exc()
                
        for subdir, dirs, files in os.walk(src):
            for fname in files:
                if ('sub_OLCI' in fname) or ('xfdumanifest' in fname):
                    continue
                else:
                    os.remove(os.path.join(subdir, fname))

if __name__=='__main__':
    # Bounding Box
	# NORTH SEA
    # xmin = -3.114326518149802 #-1.9810550389645496
    # xmax = 9.111103226811604 #8.4774877793528
    # ymin = 51.100603396036405 #51.90952058424912
    # ymax = 58.2118731386104 #58.200767268421316
	
	# MEDITERRANEAN test
	# xmin = 0.70
	# xmax = 4.90
	# ymin = 36.00
	# ymax = 41.80
	
	# GULF STREAM
	xmin = -80.00
	xmax = -65.00
	ymin = 31.00
	ymax = 40.00
    
	bound = [ymin, ymax, xmin, xmax]
	products = ['SRAL']
	
	for sensor in products:
		
		if sensor == 'SRAL':
			# In and out file directories
			src = 'C:\\Users\\vlachos\\Desktop\\SRAL\\'
			dst = 'C:\\Users\\vlachos\\Desktop\\SRAL\\'
		elif sensor == 'SLSTR':
			# In and out file directories
			src = 'C:\\Users\\vlachos\\Desktop\\SLSTR\\'
			dst = 'C:\\Users\\vlachos\\Desktop\\SLSTR\\'
		elif sensor == 'OLCI':
			# In and out file directories
			src = 'C:\\Users\\vlachos\\Desktop\\OLCI\\'
			dst = 'C:\\Users\\vlachos\\Desktop\\OLCI\\'
		
		
		nc = ncSubset_save(src, dst, bound, sensor, rmv=False)
        