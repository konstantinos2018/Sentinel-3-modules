# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 17:46:57 2018

@Description: This module contains functions which are used to read and subset
    the SRAL, SLSTR and OLCI products, after the download and first-hand
    subsetting.
@author: Kostas Vlachos
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import pdb
# My Modules
import S3coordtran as s3ct
#import S3plots

def sral_read_nc_1Hz(path, varName):
    """
    Read variables of the SRAL product that have been acquired in the 1Hz mode
    C band). The 1Hz mode denotes the lower along-track spatial resolution SRAL variables.
    
    Args:
        path = string with the path of the SRAL netCDF file
        varName = list of strings with the names of the variables that the user needs
    Returns:
        lon = 1-d ndarray with longitude of each observation of the track
        lat = 1-d ndarray with latitude of each observation of the track
        var = dictionary with the variables {varName1: ndarray, varName2: ndarray...}
        flags = dictionary with the flags that correspond to each variable
                {flagname1: 1-d ndarray, flagname2: 1-d ndarray,...}
    """
    # Read Data
    # Open NetCDF
    nc = Dataset(path, mode='r')
    
    # Read variables from ncdf
    lat = nc.variables['lat_01_C'][:].data
    lon = nc.variables['lon_20_ku'][:].data
    
    # Put variables in dictionary
    var = {}
    for name in varName:
        if name != 'flags':
            var[name] = nc.variables[name][:]
    # Initialize flags dictionary
    flags = {}
    # Put flags in dictionary
    if 'flags' in varName:
        
#        flags['surf_class_01'] = nc.variables['surf_class_01'][:]
#        flags['surf_class_20_c'] = nc.variables['surf_class_20_c'][:]
        flags['surf_class_20_ku'] = nc.variables['surf_class_20_ku'][:].data
#        flags['surf_type_01'] = nc.variables['surf_type_01'][:]
#        flags['surf_type_20_c'] = nc.variables['surf_type_20_c'][:]
        flags['surf_type_20_ku'] = nc.variables['surf_type_20_ku'][:].data
        flags['surf_type_class_20_ku'] = nc.variables['surf_type_class_20_ku'][:].data
#        flags['rad_surf_type_01'] = nc.variables['rad_surf_type_01'][:]
#        flags['dist_coast_01'] = nc.variables['dist_coast_01'][:]
#        flags['dist_coast_20_c'] = nc.variables['dist_coast_20_c'][:]
#        flags['dist_coast_20_ku'] = nc.variables['dist_coast_20_ku'][:]
#        flags['rain_flag_01_ku'] = nc.variables['rain_flag_01_ku'][:]
#        flags['rain_flag_01_plrm_ku'] = nc.variables['rain_flag_01_plrm_ku'][:]
        
    # Close netcdf
    nc.close()
    
    return lon, lat, var, flags


def sral_read_nc(path, varName):
    """
    Read variables of the SRAL product that have been acquired in the 20Hz mode
    (Ku band). The Ku band mode denotes the larger along-track spatial
    resolution SRAL variables.

    Args:
        path = string with the path of the SRAL netCDF file
        varName = list of strings which correspond to the variable names that
                  the user needs
    Returns:
        lon = 1-d ndarray with longitude of each observation of the track
        lat = 1-d ndarray with latitude of each observation of the track
        var = dictionary with the variables {varName1: ndarray, varName2: ndarray...}
        flags = dictionary with the flags that correspond to each variable
            {flagname1: 1-d ndarray, flagname2: 1-d ndarray,...}
    """
    # Read Data
    # Open NetCDF
    nc = Dataset(path, mode='r')
    
    # Read variables from ncdf
    lat = nc.variables['lat_20_ku'][:].data
    lon = nc.variables['lon_20_ku'][:].data
    
    # Put variables in dictionary
    var = {}
    for name in varName:
        if name != 'flags':
            var[name] = nc.variables[name][:]
    # Initialize flags dictionary
    flags = {}
    # Put flags in dictionary
    if 'flags' in varName:
        
#        flags['surf_class_01'] = nc.variables['surf_class_01'][:]
#        flags['surf_class_20_c'] = nc.variables['surf_class_20_c'][:]
        flags['surf_class_20_ku'] = nc.variables['surf_class_20_ku'][:].data
#        flags['surf_type_01'] = nc.variables['surf_type_01'][:]
#        flags['surf_type_20_c'] = nc.variables['surf_type_20_c'][:]
        flags['surf_type_20_ku'] = nc.variables['surf_type_20_ku'][:].data
        flags['surf_type_class_20_ku'] = nc.variables['surf_type_class_20_ku'][:].data
#        flags['rad_surf_type_01'] = nc.variables['rad_surf_type_01'][:]
#        flags['dist_coast_01'] = nc.variables['dist_coast_01'][:]
#        flags['dist_coast_20_c'] = nc.variables['dist_coast_20_c'][:]
#        flags['dist_coast_20_ku'] = nc.variables['dist_coast_20_ku'][:]
#        flags['rain_flag_01_ku'] = nc.variables['rain_flag_01_ku'][:]
#        flags['rain_flag_01_plrm_ku'] = nc.variables['rain_flag_01_plrm_ku'][:]
        
    # Close netcdf
    nc.close()
    
    return lon, lat, var, flags


def ja3_read_nc(path):
    """
    Read Jason-3 data such as lat, lon, range, alt, surface type etc.
    ***NOTE*** This function is not used
    
    Input:
        path = 
    Output:
        lon = 
        lat = 
        ssh = 
        watermask = 
        mss = 
    """
    # Read Data
    # Open NetCDF
    nc = Dataset(path, mode='r')
    
    # Read variables from ncdf
    lat = nc.variables['lat_20hz'][:][:, 0].data
    lon = nc.variables['lon_20hz'][:][:, 0].data
    altrange = nc.variables['range_20hz_ku'][:]
    altitude = nc.variables['alt_20hz'][:]
    surftypes = nc.variables['surface_type'][:]
    mss = nc.variables['mean_sea_surface'][:]
    
    # Create land surface mask
    watermask = (surftypes == 0)
    ssh = altitude - altrange
    ssh = ssh[:, 0]
    # Close netcdf
    nc.close()
    
    return lon, lat, ssh, watermask, mss


def slstr_read_nc(path):
    """
    Read SLSTR data such as lat, lon, sst
    
    Args:
        path = string with the path and filename of the SLSTR product
    Returns:
        lon = 1-d ndarray with longitude of each observation
        lat = 1-d ndarray with latitude of each observation
        sst = 1-d ndarray with the observations (i.e. SST)
    """
    # Read Data
    # Open NetCDF
    nc = Dataset(path, mode='r')
    
    # Read variables from ncdf
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    sst = nc.variables['sea_surface_temperature'][:]
    
    # Close netcdf
    nc.close()
    
    return lon, lat, sst


def olci2D_read_nc(path_coord, path_var):
    """
    Read OLCI original file data

    Args:
        path_coord = path to geo_coordinates.nc file
        path_var = path to variable nc file (e.g. chl_nn.nc)
    Returns:
        lon = 2D array of longitude
        lat = 2D array of latitude
        var = 2D array of variable of interest
    """
    # Read Data
    # Open NetCDF
    nc_latlon = Dataset(path_coord, mode='r')
    nc_var = Dataset(path_var, mode='r')
    
    # Read variables from ncdf
    lat = nc_latlon.variables['latitude'][:]
    lon = nc_latlon.variables['longitude'][:]
    var = nc_var.variables['CHL_NN'][:]
    
    # Close netcdf
    nc_latlon.close()
    nc_var.close()
    
    return lon, lat, var


def olci1D_read_nc(pathin, varName):
    """
    Read OLCI subset file data
    
    Args:
        pathin = path to subset nc file
        varName = list with strings of the variables names that are wanted as
                  outputs. If varName = ['ALL_VARS'] then all variables will be
                  as outputs.
    Returns:
        lon = 1-d ndarray of longitude
        lat = 1-d ndarray of latitude
        var = dictionary with keys=variable name and values=1D array of variable
        wqsf = 1-d array of WQSF flags etc.
    """
    
    kickout = ['latitude', 'longitude', 'WQSF']
    # Kick-out latitude and longitude variable names
    varName = list(set(varName) - set(kickout))
    
    # Open NetCDF
    nc = Dataset(pathin)
    # Read latitude, longitude and flags
    lat = nc.variables['latitude'][:]
    lon = nc.variables['longitude'][:]
    flags = nc.variables['WQSF'][:]
    
    # Initialize output variables dictionary
    var = {}
    
    try:
        # Check if variables names is only 'ALL_VARS'
        if (len(varName)==1) & (varName[0]=='ALL_VARS'):
            for v in nc.variables:
                var[v] = nc.variables[v][:]
        else:
            for name in varName:
                var[name] = nc.variables[name][:]
    except Exception as e:
        print(str(e))

    return lon, lat, var, flags


def slstr1D_read_nc(pathin, varName):
    """
    Read SLSTR subset file data
    
    Args:
        pathin = path to subset nc file
        varName = list with strings of the variables names that are wanted as
                  outputs. If varName = ['ALL_VARS'] then all variables will be
                  as outputs.
    Returns:
        lon = 1D array of longitude
        lat = 1D array of latitude
        var = dictionary with keys=variable name and values=1D array of variable
    """
    kickout = ['lat', 'lon', 'l2p_flags', 'quality_level']
    # Kick-out latitude and longitude variable names
    varName = list(set(varName) - set(kickout))
    
    # Open NetCDF
    nc = Dataset(pathin)
    # Read latitude, longitude and flags
    lat = nc.variables['lat'][:]
    lon = nc.variables['lon'][:]
    l2p_flags = nc.variables['l2p_flags'][:]
    quality_level = nc.variables['quality_level'][:]
    
    # Initialize output variables dictionary
    var = {}
    
    try:
        # Check if variables names is only 'ALL_VARS'
        if (len(varName)==1) & (varName[0]=='ALL_VARS'):
            for v in nc.variables:
                var[v] = nc.variables[v][:]
        else:
            for name in varName:
                var[name] = nc.variables[name][:]
    except Exception as e:
        print(e)

    return lon.data, lat.data, var, l2p_flags.data, quality_level.data


def sral_subset_nc(lon, lat, var, flags, bound):
    """
    Subsets the SRAL product
    
    Args:
        lon = 1-d ndarray with longitude of each observation of the track
        lat = 1-d ndarray with latitude of each observation of the track
        var = dictionary with the variables {varName1: ndarray, varName2: ndarray...}
        flags = 
        bound = list withe bounding box coordinates [xmin, xmax, latmin, latmax]
    Returns:
        lon = 1-d ndarray with longitude of each observation of the track
        lat = 1-d ndarray with latitude of each observation of the track
        var = dictionary with the variables {varName1: ndarray, varName2: ndarray...}
        flags = dictionary with the flags that correspond to each variable
            {flagname1: 1-d ndarray, flagname2: 1-d ndarray,...}
    """
#    # Check if lon, lat, flags are masked
#    if ma.is_masked(lon):
#        lon = lon.data
#    if ma.is_masked(lat):
#        lat = lat.data
#    for key in flags.keys():
#        if ma.is_masked(flags[key]):
#            print(ma.is_masked(flags[key]))
#            flags[key] = flags[key].data
    
    # pass variables
    lonmin = bound[0]
    lonmax = bound[1]
    latmin = bound[2]
    latmax = bound[3]
    

    ssha = var['ssha_20_ku']
    #    varout = {}
    for keys in var.keys():
#        if lon.shape == var[key].data.shape:
        var[keys] = var[keys].data[np.logical_not(ssha.mask)]
    # Apply mask
    lon = lon[np.logical_not(ssha.mask)]
    lat = lat[np.logical_not(ssha.mask)]
    for keys in flags.keys():
#        if flags[keys].shape == lon.shape:
        flags[keys] = flags[keys][np.logical_not(ssha.mask)]
    ssha = ssha.data[np.logical_not(ssha.mask)]

    # Create spatial mask
    spatialmask = (lat > latmin) & (lat < latmax) & (lon > lonmin) & (lon < lonmax)
    # Apply spatial mask
    lat = lat[spatialmask]
    lon = lon[spatialmask]
    ssha = ssha[spatialmask]
    for keys in var.keys():
        var[keys] = var[keys][spatialmask]
    for keys in flags.keys():
        flags[keys] = flags[keys][spatialmask]
    
    # Return lon, lat and var subset
    return lon, lat, var, flags

def ja3_subset_nc(lon, lat, var, watermask, bound):
    """
    Jason-3 subset
    ***NOTE*** This function is not used
    """ 
    # pass variables
    lonmin = bound[0]
    lonmax = bound[1]
    latmin = bound[2]
    latmax = bound[3]
    
    #---- subset -----
    lonm = (lon > lonmin) & (lon < lonmax)
    latm =(lat > latmin) & (lat < latmax)
    idxcoord = lonm & latm
    
    # Create mask for valid var values and water surface
    idxvar = np.logical_not(var.mask)
    idxvarwater = idxvar & watermask
    
    # Create final mask of valid values
    idx = idxcoord & idxvarwater
    
    # Create new lon, lat, var based on mask
    lon = lon[idx]
    lat = lat[idx]
    var = var.data[idx]
    
    # Return lon, lat and var subset
    return lon, lat, var


def slstr_olci_subset_nc(lon, lat, var, bound):
    """
    Take a spatial and valid value subset of lan, lot, var based on bounds
    
    Args:
        lon = 1d array of longitude
        lat = 1d array of latitude
        var = dictionary with keys=variable name and values=1D array of masked variable
        flags = 
        bound = list with spatial boundaries [xmin, xmax, ymin, ymax]
    Returns:
        lon = 1d masked array of longitude
        lat = 1d masked array of longitude
        newVar = dictionary with keys=variable name and values=1d masked array of variable
    """
    # pass variables
    lonmin = bound[0]
    lonmax = bound[1]
    latmin = bound[2]
    latmax = bound[3]
    
    # Create spatial mask
    idxcoord = (lon > lonmin) & (lon < lonmax) & (lat > latmin) & (lat < latmax)
    
    # Initialize variable dictionary
    newVar = {}
            
    for name in var.keys():
        # Copy/paste 'time' variable if there is in var dictionary
        if name == 'time':
            newVar[name] = var[name].data
        else:
            # Create mask for valid var values (False for FillValues)
            idxvar = np.logical_not(var[name].mask)
        
            # Create final mask of valid values
            idx = np.logical_and(idxcoord, idxvar)
        
            # Create new var based on mask
            newVar[name] = np.ma.masked_where(idx, var[name].data)

    return newVar

if __name__ == '__main__':
#    path = 'D:\\vlachos\\DOCUME~1\\KVMSCT~1\\Data\\SATELL~1\\TESTDA~1\\S3-A\\201804~1\\S3A_SL~1\\S3A_SL_2_WST____20180419T194008_20180419T212108_20180421T040631_6059_030_170______MAR_O_NT_003.SEN3\\'
#    fileName = '20180419194008-MAR-L2P_GHRSST-SSTskin-SLSTRA-20180421040631-v02.0-fv01.0.nc'
    path = 'D:\\vlachos\\DOCUME~1\\KVMSCT~1\\Data\\SATELL~1\\TESTDA~1\\S3-A\\201804~1\\S3A_SR~1\\S3A_SR_2_LAN____20180419T102520_20180419T111512_20180514T220143_2991_030_165______LN3_O_NT_003.SEN3\\'
    fileName = 'enhanced_measurement.nc'
    inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
    outEPSG = '+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs ' # North Sea
#    path = 'D:\\vlachos\\Documents\\KV MSc thesis\\Data\\Satellite\\Test data\\S3-A\\201804~1\\S3A_OL~1\\S3A_OL_2_WFR____20180420T100608_20180420T100908_20180421T155944_0179_030_179_1980_MAR_O_NT_002.SEN3\\'
#    fileName = 'geo_coordinates.nc'
#    fileName2 = 'chl_oc4me.nc'
    shppath = 'D:\\vlachos\\Documents\\KV MSc thesis\\Data\\Country_borders\\North_Sea_BorderCountries_3035.shp'
#    shppath4326 = 'D:\\vlachos\\Documents\\KV MSc thesis\\Data\\Country_borders\\North_Sea_BorderCountries.shp'
    bound = [3500000, 4300000, 3100000, 4000000]
#    bound = [50.7, 60.05, -2, 9.78]
    lon, lat, ssha, watermask = sral_read_nc(path+fileName)
    x, y = s3ct.sral_coordtran(lon, lat, inEPSG, outEPSG)
#    x, y, ssha = sral_subset_nc(x, y, ssha, watermask, bound)
#    S3plots.sral_scatter(x, y, ssha, shppath, bound)
#    None