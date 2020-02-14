# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:56:35 2018

DESCRIPTION:
    The module contains functions that are used in the post-processing of 
    the S3 products after the pre-processing/preparation stage
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import scipy as sp
import scipy.interpolate as spint
import numpy as np
from netCDF4 import Dataset
from scipy import spatial
import pdb


def ckdnn_traject_idw(xsral, ysral, xvar, yvar, var, kwargs):
    """
    Interpolates using inverse distance weighted inteprolation
    
    Args:
        xsral = 1D array of x coordinates of altimetry track
        ysral = 1D array of y coordinates of altimetry track
        xvar = 1D array of x coordinates of some variable (e.g. sst)
        yvar = 1D array of y coordinates of some variable (e.g. sst)
        var = 1D array of the known values of the variable of interest
        kwargs = keyword arguments of the tree.query() method
    Returns:
        var_est = 1D array of estimated values of some variable at positions (xsral, ysral)
    """
    # Check input size
    if var.size == 0:
        return None
                        
    # Stack x and y coordinates of SRAL and variable
    xysralstack = np.dstack([xsral, ysral])[0]
    xyvarstack = np.dstack([xvar, yvar])[0]
    
    tree = spatial.cKDTree(xyvarstack)
    out = tree.query(xysralstack, **kwargs)
    
    # Assign distances to variable and indices to variable
    dist = out[0]
    indices = out[1]
    # find INF positions in distances matrix
    mask_inf = np.isinf(dist)
    # Give nan values to INF distances
    dist[mask_inf] = np.nan
    
#    # REDEFINE dist, indices and mask_inf in order to make
#    try:
#        print(np.where(dist < 1101)[1].max())
#    except:
#        pass
    # Compute variable matrix
    # Initialize. var_matrix has shape (N, k), where N=number of SRAL points along
    # the trajectory and k=number of nearest neighbors
    var_matrix = np.zeros(indices.shape)
    for i in range(indices.shape[1]):
        temp_mask = np.logical_not(mask_inf[:, i])
        var_matrix[temp_mask, i] = var[indices[temp_mask, i]]
  

    # Compute weights based on distances
    # initialize
    weight = np.zeros(dist.shape)
    
    # Take useful values indices (ommiting the inf/nans)
    idx = np.logical_not(mask_inf)
    # Compute weight
    weight[idx] = 1/dist[idx]
    # Check for zero distances      
    for i in range(weight.shape[0]):
        zero_mask = dist[i, :] == 0       
        # Put unit(1) weight if the distance is zero and zero (0) weight
        # to the rest of the neighbors. This is used if the query point falls on exactly
        # one of the observation points
        if np.any(zero_mask) == True:
            weight[i, zero_mask] = 1
            weight[i, ~zero_mask] = 0
            
    # Compute sum of weights
    sum_weight = weight.sum(axis=1)
    # Compute nominator
    nom = (weight*var_matrix).sum(axis=1)
    
    # Compute estimates
    var_est = nom/sum_weight
        
    return var_est

def twoDirregularFilter(xsral, ysral, var_interp, xvar, yvar, var, kwargs):
    """
    Filters a 2D variable that is given in an 1D array (x, y, value) using the
    moving average method.
    
    Args:
        xsral = 1D array of SRAL x cartesian coordinates of query points
        ysral = 1D array of SRAL y cartesian coordinates of query points
        var_interp = 1D array of variable values on positions (xsral, ysral)
            along-track
        xvar = 1D array of x cartesian coordinates of query points
        yvar = 1D array of y cartesian coordinates of query points
        var = 1D array of variable values at positions (xvar, yvar) which
            which are around (xsral, ysral)
    Returns:
        var_movAv = 1D array of estimated values of the variable at positions
        (xsral, ysral)
    """
    # Check input size
    if (var.size == 0) or (var_interp.size == 0):
        return None
    
    # *Note* The data are split and processed into segments in order to avoid
    # memory issues
    
    N = 500 # Number of segments
    XY_query = np.dstack([xsral, ysral])[0]
    # control the size of the segments to be split into
    if xsral.size <= N:
        N = 1
    XY_query = np.array_split(XY_query, N) # Segment in N sub-arrays
    XY_obs = np.dstack([xvar, yvar])[0]
    tree_obs = spatial.cKDTree(XY_obs) # tree of observations (var [e.g. sst])
    
    var_movAv = [] # Initialization
#    i=0 # counter
    for segment in XY_query:
#        i = i + 1
#        print(i)
        tree_query = spatial.cKDTree(segment) # tree of query points
        out = tree_query.query_ball_tree(tree_obs, **kwargs) # find nearest neighbours indices
        
        # Compute moving average
        for point in out:
            var_movAv.append(np.nanmean(var[point]))
#        pdb.set_trace()
    
    # convert from list to numpy array
    var_movAv = np.asarray(var_movAv)
    
    return var_movAv


def ckdnn_traject_knn(xsral, ysral, xvar, yvar, var, query_args):
    """
    k-NN interpolation and moving average filter
    Input:
    Output:
    """    
    # Stack x and y coordinates of SRAL and variable
    xysralstack = np.dstack([xsral, ysral])[0]
    xyvarstack = np.dstack([xvar, yvar])[0]
    
    tree = spatial.cKDTree(xyvarstack)
    # Explanation of out variable:
    #   out[0] represents distances and is NxM where N=number of SRAL points and M=number of k-NNs
    #   out[1] represents indices of SST variable (or other that is interpolated) and has the same
    #          size as above
    out = tree.query(xysralstack, **query_args)
    
    # Assign distances to variable and indices to variable
    dist = out[0]
#    indices = out[1]
#    # find INF positions in distances matrix
    mask_inf = np.isinf(dist)
    # Give nan values to INF distances
    dist[mask_inf] = np.nan
    # Interpolate
#    var_est =var[indices[0]]
    
    return (dist, out[1])


def rescale_between(var, ub, lb):
    """
    Rescale ndarray of values between a given range (i.e. min-max normalization)
    
    Args:
        var = 1D array of values
        ub = upper bound of the rescaled range
        lb = lower bound of the rescaled range
    Returns:
        var_res = 1D array of the rescaled values
    """    
    # Check if input var is empty
    if var.size == 0:
        return var
    # Check
    if np.all(np.isnan(var)) == True:
        return var
    
    # Define min and max of the given array ommitting the nans
    varmin = np.amin(var[np.logical_not(np.isnan(var))])
    varmax = np.amax(var[np.logical_not(np.isnan(var))])
    
    # Compute rescaled input array
    var[np.logical_not(np.isnan(var))] = lb + ((var[np.logical_not(np.isnan(var))] - varmin)/(varmax - varmin))*(ub - lb)
    
    return var


def sral_dist(xsr, ysr):
    """
    Compute cumulative distance of the altimetry ground track. Be careful
    from which direction the altimetry ground track begins. If it is an ascending
    node, then the 1D array will start from the South position towards the North
    position. If it is a descending node, the the 1D array will start from the
    North position towards the South position.
    
    Args:
        xsr = 1D array of x coordinates of SRAL
        ysr = 1D array of y coordinates of SRAL
    Returns:
        dist = 1D array of cumulative distance ranging between 0 and N meters
    """
    # Compute distances between SRAL points
    dstx = np.diff(xsr, axis=0)
    dsty = np.diff(ysr, axis=0)
    
    # Compute Euclidean distance
    dist = np.sqrt(dstx**2 + dsty**2)
    
    # Append 0 value as the beginning of the trajectory
    dist = np.concatenate(([0], dist))
    
    # Compute cumulative sum of distances
    dist = np.cumsum(dist)
    
    # Return cumulative distance
    return dist


def sral_dist_nans(dist, var, threshold):
    """
    Inserts NaN in-between the cumulative distance elements that is an output
    of the sral_dist(). The purpose of this function is to prepare the tracks
    for cross-section (spaceseries) visualization.
    
    Args:
        dist = 1D array of cumulative distance starting with 0
        var = 1D array of the variable which also will have nans inserted
        threshold = 1 numeric number of the threshold distance between two points
    Returns:
        dist_nan = 1D array of cumulative distance with NaN element at the places
        where the distance between two elements (points) is greater than the given
        threshold
        var_nan = 1D array of variable with NaNs inserted
        idx = 1D array with the element positions of the dist array where the
        NaNs should be inserted
    """
    # Convert to float32 (essential for using NaNs in ndarray in numpy version >13)
#    dist = dist.astype('float32')
    
    # Find differences
    dist_diff = np.diff(dist)
    
    # Concatenate 0 in the beginning
    dist_diff = np.concatenate(([0], dist_diff))
    
    # Find distances of threshold
    dist_diffsum = dist_diff > threshold
    
    # create ndarray of indices
    idx = np.asarray(range(dist.size))
    
    # select the indices to be used to insert nans
    idx = idx[dist_diffsum]
    
    # create nan vector of idx size
    idx_nan = idx*np.nan
    
    # Insert nans
    dist_nan = np.insert(dist, idx, idx_nan)
    var_nan = np.insert(var, idx, idx_nan)
    
    # Return cumulative distance with NaNs
    return dist_nan, var_nan, idx


def extract_bits(x,num_bits):
    """
    Extract the pixel values to bit values. For example, if variable x = 3 and 8-bit type
    then the corresponding bit values are [1 1 0 0 0 0 0 0] with the bit numbers being
    [1 2 3 4 5 6 7 8] = [2^0 2^1 2^2 2^3 2^4 2^5 2^6 2^7]. If x is a numpyarray
    x = [3, 2] and 8-bit type then the corresponding bit values are
    [[1 1 0 0 0 0 0 0], [0 1 0 0 0 0 0 0]]
    E.g. see https://forum.step.esa.int/t/sentinel-3-qulity-flags-usage/10351/2
    
    Args:
        x = 1D array with the bit values
        num_bits = numeric type of array x (e.g. 8-bit, 16-bit, 32-bit).
                   In the case of Sentinel-3 the type is 64-bit
    Returns:
        out = 2D boolean array (num_of_pixels x bit_number)
    """
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    flag_vals = 2**np.arange(num_bits, dtype=''.join(('uint',str(num_bits)))).reshape([1,num_bits])
    new_shape = xshape
    new_shape.append(num_bits)
  
    # Split into segments
    N = 5
    x_seg = np.array_split(x, N)
  
    # Initialize the first element
    out = (x_seg[0] & flag_vals).astype(np.bool)
    # Repeat for every element except for the 1st
    for segment in x_seg[1:]:
        temp = (segment & flag_vals).astype(np.bool)
        out = np.concatenate((out, temp))
#   pdb.set_trace()
    out = out.reshape(new_shape)
  
    return out

#def extract_mask(bitval, ndinput):
#    # check whether there is a certain bit value in every pixel
#    # Take the number of pixels of the input array
#    npixels = ndinput.shape[0]
#    # Initialize dictionary and 1D array
#    ndoutput = {}
#    
#    for key, val in bitval.items():
#        # Initialize 1D array of mask
#        ndoutput[key] = np.zeros(npixels, dtype=np.bool)
#        for i in range(npixels):
#            if np.isin(val, ndinput[i, :]):
#                ndoutput[key][i] = True
#    
#    return ndoutput


def extract_mask(bitval, ndinput, num_bits):
    """
    Creates numpy arrays with boolean masks with respect to the flags. The numpy
    arrays are written as values in a dictionary whose keys are the flags names (meanings)
    
    Args:
        bitval = dictionary (key=flag name, value=flag value)
        ndinput = 1D array with extracted bit using extract_bits() function
    Returns:
        ndoutput = dictionary (key=flag name, value=1D boolean)
    """
    # Initialize dictionary
    ndoutput = {}
    for key, val in bitval.items():
        for i in range(num_bits):
            if 2**i == val:
                ndoutput[key] = ndinput[:, i]
            else:
                continue

    return ndoutput


def extract_maskmeanings(pathfile):
    """
    Extract the meanings of each flag
    
    Args:
        pathfile = path to the netcdf file which includes the flags (OLCI or SLSTR)
    Returns:
        varout = dictionary with keys=mask meanings and values=mask values (bit numbers)
    """
    # Open netCDF
    nc = Dataset(pathfile)
    
    # check whether it is OLCI or SLSTR
    if 'WQSF' in nc.variables.keys():
        # Open WQSF metadata
        flag_dic = nc.variables['WQSF']
        
        # Assign flag meaning to variable
        flag_meanings = flag_dic.getncattr('flag_meanings')
        # convert unicode to string
        flag_meanings = [str(i) for i in flag_meanings]
        flag_meanings = ''.join(flag_meanings)
        flag_meanings = flag_meanings.split(' ')
        # Assign flag masks to variable
        flag_masks = flag_dic.getncattr('flag_masks')
        
        # Create dictionary
        varout = dict(zip(flag_meanings, flag_masks))
        
        nc.close()
        
        return varout
    
    elif ('l2p_flags' in nc.variables.keys()) or ('quality_level' in nc.variables.keys()):
        # Open l2p_flags metadata
        flag_dic = nc.variables['l2p_flags']
        # Assign flag meaning to variable
        flag_meanings = flag_dic.getncattr('flag_meanings')
        # convert unicode to string
        flag_meanings = [str(i) for i in flag_meanings]
        flag_meanings = ''.join(flag_meanings)
        flag_meanings = flag_meanings.split(' ')
        # Assign flag masks to variable
        flag_masks = flag_dic.getncattr('flag_masks')
        
        varout_1 = dict(zip(flag_meanings, flag_masks))
        
        # Open quality_level metadata
        flag_dic = nc.variables['quality_level']
        # Assign flag meaning to variable
        flag_meanings = flag_dic.getncattr('flag_meanings')
        # convert unicode to string
        flag_meanings = [str(i) for i in flag_meanings]
        flag_meanings = ''.join(flag_meanings)
        flag_meanings = flag_meanings.split(' ')
        # Assign flag masks to variable
        flag_masks = flag_dic.getncattr('flag_values')
        
        varout_2 = dict(zip(flag_meanings, flag_masks))
        
        nc.close()
        
        return varout_1, varout_2


def apply_masks_olci(inpvar, varname, inpmasks):
    """
    Apply masks to OLCI variables. The user can adjust the values of each flag
    according to preference
    
    Args:
        inpvar = 1D array of the given variable
        varname = string of the name of the variable. The name is the same as given
                  in the netCDF
        inpmasks = dictionary of the masks (output of S3postproc.extract_mask() function)
    Returns:
        outvar = 1D array of the given variable with masks applied
        outmask = 1D boolean array of the sum of the relevant masks
    """
    # Create products' name lists
    prod_algopen = ['CHL_OC4ME', 'CHL_OC4ME_err']
    prod_diffus = ['KD490_M07', 'KD490_M07_err']
    prod_par = ['PAR', 'PAR_err']
    prod_aer = ['T865', 'A865', 'A865_err']
    prod_algcomp = ['CHL_NN', 'CHL_NN_err']
    prod_susp = ['TSM_NN', 'TSM_NN_err']
    prod_absorp = ['ADG443_NN', 'ADG443_NN_err']
    prod_iwd = ['IWV', 'IWV_err']
    
    # Check if in water products
    if varname in (
            prod_algopen + prod_diffus + prod_par + prod_aer + prod_algcomp 
            + prod_susp + prod_absorp
            ):
        # Create 1st level mask
        outmask = (
                inpmasks['WATER']
                & np.logical_not(inpmasks['CLOUD']) 
                & np.logical_not(inpmasks['CLOUD_AMBIGUOUS'])
                & np.logical_not(inpmasks['CLOUD_MARGIN'])
                & np.logical_not(inpmasks['INVALID'])
                & np.logical_not(inpmasks['COSMETIC'])
                & np.logical_not(inpmasks['SATURATED'])
                & np.logical_not(inpmasks['SUSPECT'])
#                & np.logical_not(inpmasks['HISOLZEN'])
                & np.logical_not(inpmasks['HIGHGLINT'])
                & np.logical_not(inpmasks['SNOW_ICE'])
                )
        # Check if in Open Water Products
        if varname in (prod_algopen + prod_diffus + prod_par + prod_aer):
            outmask = (
                    outmask & np.logical_not(inpmasks['AC_FAIL'])
                    & np.logical_not(inpmasks['WHITECAPS'])
                    & np.logical_not(inpmasks['ANNOT_ABSO_D'])
                    & np.logical_not(inpmasks['ANNOT_MIXR1'])
                    & np.logical_not(inpmasks['ANNOT_DROUT'])
                    & np.logical_not(inpmasks['ANNOT_TAU06'])
                    & np.logical_not(inpmasks['RWNEG_O2'])
                    & np.logical_not(inpmasks['RWNEG_O3'])
                    & np.logical_not(inpmasks['RWNEG_O4'])
                    & np.logical_not(inpmasks['RWNEG_O5'])
                    & np.logical_not(inpmasks['RWNEG_O6'])
                    & np.logical_not(inpmasks['RWNEG_O7'])
                    & np.logical_not(inpmasks['RWNEG_O8'])
                    )
#            # Check which product exactly
            if varname in prod_algopen:
                outmask = outmask & np.logical_not(inpmasks['OC4ME_FAIL'])
            elif varname in prod_diffus:
                outmask = outmask & np.logical_not(inpmasks['KDM_FAIL'])
            elif varname in prod_par:
                outmask = outmask & np.logical_not(inpmasks['PAR_FAIL'])
#        # If Complex Water Products
        elif varname in (prod_algcomp + prod_susp + prod_absorp):
            outmask = outmask & np.logical_not(inpmasks['OCNN_FAIL'])
    
    # Apply mask
    outvar = inpvar[outmask]
    
    return (outvar, outmask)


def apply_masks_slstr(inpvar, varname, inpmasks, quality_level):
    """
    Apply masks to SST variable. The user can adjust the values of each flag
    according to preference
    
    Args:
        inpvar = 1D array of the given variable
        varname = string of the name of the variable. The name is the same as given
                  in the netCDF
        inpmasks = dictionary of the masks (output of S3postproc.extract_mask() function)
                   In specific, the output refers to the l2p_flags which contains
                   flags, coded as bit numbers
    Returns:
        outvar = 1D array of the given variable with masks applied
        outmask = 1D boolean array of the sum of the relevant masks
    """
    # Derive the ocean measurements
    # bits from 0-5
    outmask = (
            np.logical_not(inpmasks['land'])
            & np.logical_not(inpmasks['microwave'])
            & np.logical_not(inpmasks['ice'])
            & np.logical_not(inpmasks['lake'])
            & np.logical_not(inpmasks['river'])
            )
    # bits greater than 5        
    outmask = (
            outmask & np.logical_not(inpmasks['exception'])
            & np.logical_not(inpmasks['cloud'])
            & np.logical_not(inpmasks['sun_glint'])
            )
    
    # Apply quality_level masks. >=2 is considered to depict valid values
    outmask = outmask & (quality_level >= 2)
    
    # Apply final mask to input variable
    outvar = inpvar[outmask]
    
    return (outvar, outmask)


def apply_masks_sral(inpvar, varname, flags):
    """
    Apply masks to SRAL variable. The user can adjust the values of each flag
    according to preference
    
    Args:
        inpvar = 1D array of variable
        varname = string of the variable name
        inpmasks = dictionary with flags which is output of nc_manipul.sral_read_nc()
    Returns:
        outvar = 1D array of variable with mask applied
        outmask = 1D boolean array of mask
    """
    # True if equals to 0 (see flag meanings for meaning of 0)
    flags['surf_type_20_ku'] =  flags['surf_type_20_ku'] == 0
    flags['surf_class_20_ku'] = flags['surf_class_20_ku'] == 0
    flags['surf_type_class_20_ku'] = flags['surf_type_class_20_ku'] == 0
#    flags['range_ocean_qual_20_ku'] = flags['range_ocean_qual_20_ku'] == 0 # YES (meaning)
    # Surface type flag
    outmask = (
            flags['surf_type_20_ku']
            & flags['surf_class_20_ku']
            & flags['surf_type_class_20_ku']
#            & flags['range_ocean_qual_20_ku']
            )
    outvar = inpvar[outmask]
    
    return outvar, outmask


def check_npempty(var):
    """
    Check if input array is empty
    
    Args:
        var = np array
    Returns:
        True = array is empty
        False = array is not empty
    """
    if var.size == 0:
        return True
    else:
        return False

