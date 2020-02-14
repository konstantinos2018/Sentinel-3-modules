# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:25:06 2019
This module contains various functions that are used in the preparation and
analysis of the data before using them as inputs to the ML analysis algorithms
"""

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime as dt
from collections import OrderedDict
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os
# My Modules
import s3utilities

def my_pca(pd_input, pd_fit_data, prin_comp):
    """
    Apply PCA
    
    Args:
        pd_input = panda array input
    Returns:
        np_output = nd array output
    """
    pca = PCA(n_components=prin_comp)
#    pca.fit_transform(pd_input)
    pca.fit(pd_fit_data)
    
    np_output = pca.transform(pd_input)
    
    return np_output


def my_standardizer(input_data, fit_data):
    """
    Standardizes input ndarray/dataframe using the z-score method
    
    Args:
        input_data = nd array with data to be rescaled
        fit_data = nd array with data which the scale fit will be based on
    Returns:
        output_data = nd array with rescaled output data
    """
    #    # Check if ndarray
    if type(input_data) == np.ndarray:
        pass
    
    elif type(input_data) == pd.DataFrame:
        input_data = np.array(input_data)
        
    else:
        raise Exception('Input data is not of ndarray type')
    
    # Check how many dimensions
    if len(input_data.shape) == 2:
        pass
    else:
        raise Exception('Input data shape (input.shape) size is not equal to 2. Change shape.size')
    
    scaler = StandardScaler()
    scaler.fit(fit_data)
    
    # Rescale
    output_data = scaler.transform(input_data)
    

    return output_data


def pick_npz_dates(npz_files, start_date, n_plus):
    """
    Picks specific files based on dates criterion
    
    Args:
        npz_files = input list of strings that denote the names of the npz files
        start_date = string denoting the starting date (form: %Y-%m-%d e.g. 2018-12-04)
        n_plus = integer that denotes how many days after the start_date you want to take into account
    Returns:
        npz_files_out = output list of string that denote the chosen npz files
    """
    # Convert start_date to datetime
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d') #start date
    
    # Search for the given date span
    npz_files_out = []
    for npz in npz_files:
        npz_date = dt.datetime.strptime(npz[4:14], '%Y-%m-%d') # convert to datetime
        end_date = start_date + dt.timedelta(n_plus) # end date
        
        if (npz_date >= start_date) and (npz_date <= end_date):
            npz_files_out.append(npz)
    
    return npz_files_out


def split_npz_files_by_dates(npz_files, n_plus):
    """
    Split npz_files list in batches (groups) where each group represents files that fall inside the desired time span
    
    Args:
        npz_files = input list of strings that denote the names of the npz files
        n_plus = integer that denotes how many days after the start_date you want to take into account
    Returns:
        npz_files_out = output list of lists of strings that denote the chosen npz files that are grouped by dates
    """
    npz_files_out = [] # Initialize npz_files_out

    npz_files_dt = [dt.datetime.strptime(npz[4:14], '%Y-%m-%d') for npz in npz_files] # Convert to datetimes
    npz_files_dt_unique = list(OrderedDict.fromkeys(npz_files_dt)) # Delete dublicates
    
    # Create npz_files_out
    for item in npz_files_dt_unique:
        npz_files_out.append([item_npz for item_dt, item_npz in zip(npz_files_dt,npz_files) if (item_dt>=item) and (item_dt<=item+dt.timedelta(n_plus))])
    return npz_files_out


def grid_feature_matrix_from_npz(npz_file, variable_out=[]):
    """
    Creates the feature matrix for a grid
    
    Args:
        npz_file = list of npz files (paht+filename)
        variable_out = list of strings denoting the name of the variables that you
                       want to kick out
    Returns:
        matrix = feature matrix (ndarray) NxM (N=number of points (observations),
                 M=number of features (SST, OLCI etc))
    """
    # Derive names of variables
    data = np.load(npz_file, encoding='latin1', allow_pickle=True)
    # Retrieve dictionary
    data = data['arr_0'].item()
    other_variables = {'Metadata': data['Metadata'],
                       'n_x':data['n_x'],
                       'n_y':data['n_y']}
    X = data['X']
    Y = data['Y']
    del data['Metadata'], data['X'], data['Y'], data['n_x'], data['n_y']
    
    matrix = pd.DataFrame.from_dict(data, dtype=np.float32)
    return X, Y, matrix, other_variables


def feature_matrix_from_npz(npz_file):
    """
    Take the npz_file that is going to be used in training and keep distance, metadata as ndarray,
    as well as feature matrix as pd.DataFrame
    
    Args:
        npz_file = npz file which includes label and feature data matrix,
                   distance vector of the track and metadata of the product
    Returns:
        matrix = pd.DataFrame of the labels and feature matrix
        distance = array of the distanes of the track
        metadata = Metadata of the product
    """
    #    variable_out = ['Distance', 'Metadata']
#    variable_out = ['Distance', 'Metadata', '150km']
#    variable_out = ['Distance', 'Metadata', '50km', '150km']
    variable_out = ['Distance', 'Metadata', 'SSHA_901', 'SSHA_623']
    
    # Derive names of variables
    data = np.load(npz_file, encoding='latin1', allow_pickle=True)
    # Retrieve dictionary
    data = data['arr_0'].item()
    
    distance = data[variable_out[0]]
    metadata = data[variable_out[1]]
    
    variable_out = [item_2 for item_1 in variable_out for item_2 in data.keys() if item_1 in item_2]
    
    # Kick-out variable_out
    for item in variable_out:
        out = data.pop(item, None)
    del out            
    
    matrix = pd.DataFrame.from_dict(data, dtype=np.float32) # feature matrix to pd.DataFrame
    
    return matrix, distance, metadata


def imputate_nans_feature_matrix(pd_input, method, drop_nan=True):
    """
    Impute missing values of panda DataFrame
    
    Args:
        pd_input = pd.DataFrame
        method = 'Interpolate' or 'Zero' or 'Kickout'
    Returns:
        pd_output: pd.DataFrame with NaNs imputated based on the input approach
    """
    if method=='Interpolate': # 1) IMPUTATION OF NANs- INTERPOLATION AND DROP THE REST OF THE NANS
        
        # Interpolate the NAN values inside the dataset
        pd_output = pd_input.interpolate(method='akima', limit=100, limit_direction='both', axis=0)
        # Interpolate (actually extrapolate) the values at the edges
        pd_output = pd_output.interpolate(method='linear', limit=50, limit_direction='both', axis=0)
        # Detect and Delete remaining rows with NaNs
        nan_idx = pd_output.isna()
        nan_idx = nan_idx.any(axis=1)
        
        if drop_nan==True:
            pd_output = pd_output.dropna()
        else:
            pass
        
    elif method=='Zero': # 2) IMPUTATION OF NANs- ZEROS
        
        # Detect and Delete remaining rows with NaNs
        nan_idx = pd_input.isna()
        
        
        # Replace NaN with 0 in the label dataset
        pd_output = pd_input.fillna(value=0)
    
    elif method=='Kickout': # 3) DELETE ROWS WITH NANs
    
        # Delete row with NaN in the label dataset
        nan_idx = pd_input.isna()
        nan_idx = nan_idx.any(axis=1)
        
        pd_output = pd_input.dropna()
    
    elif method =='Interpolate_Random': # Imputate with random values
        # Delete row with NaN in the label dataset
        nan_idx = pd_input.isna()
        nan_idx = nan_idx.any(axis=1)
        
        # Interpolate the NAN values inside the dataset
        pd_input = pd_input.interpolate(method='akima', limit=150, limit_direction='both', axis=0)
        
        pd_output = pd.DataFrame(data=np.random.randn(pd_input.shape[0], pd_input.shape[1]),
                                       columns=pd_input.columns,
                                       index=pd_input.index)
        
        pd_output.update(pd_input)
    elif method == 'Negative':
        # Detect and Delete remaining rows with NaNs
        nan_idx = pd_input.isna()
        nan_idx = nan_idx.any(axis=1)
        
        # Replace NaN with 0 in the label dataset
        pd_output = pd_input.fillna(value=-2)
        
    else:
        raise Exception('Incorrect method argument value')
                
    return pd_output, nan_idx

def create_grid_true(x_min, x_max, y_min, y_max, grid_step):
    """
    Creates a grid that is bounded by given X, Y coordinates of two points
    (i.e. bounding box)
    Args:
        x_min = number with minimum x coordinate
        x_max = number with maximum x coordinate
        y_min = number with minimum y coordinate
        y_max = number with maximum y coordinate
        grid_step = step each point of the grid
    Returns:
        x_array = 1-d ndarray with the x coordinates
        y_array = 1-d ndarray with the y coordinates
        n_x = number of grid's points in the x direction
        n_y = number of grid's points in the y direction
    """
    # Compute number of grid points in x and y axes
    n_x = (x_max - x_min)//grid_step # x axis
    n_y = (y_max - y_min)//grid_step # y axis
    x_array = np.linspace(x_min, x_max, n_x) # x coordinates array
    y_array = np.linspace(y_min, y_max, n_y) # y coordinates array
#    index_x = np.arange(n_x) # indices of x coordinates array
#    index_y = np.arange(n_y) # indices of y coordinates array
    
    x_grid, y_grid = np.meshgrid(x_array, y_array, indexing='ij') # x, y coordinates grid
#    index_grid_x, index_grid_y = np.meshgrid(index_x, index_y, indexing='ij') # x, y indices grid
    x_array = np.matrix.flatten(x_grid) # x coordinates array of grid
    y_array = np.matrix.flatten(y_grid) # y coordinates array of grid
#    index_x = np.matrix.flatten(index_grid_x) # indices of x coordinates array of grid
#    index_y = np.matrix.flatten(index_grid_y) # indices of y coordinates array of grid

    return x_array, y_array, n_x, n_y


def  matrix_min_max_rescale(pd_input, ub, lb, axis):
    """
    Min-Max normalization on ndarray/pd.dataFrame inputs
    
    Args:
        pd_input = pd.dataFrame
        ub = upper bound values
        lb = lower bound value
        axis = the axis along which the rescaling will take place
    Returns:
        pd_output = rescaled output
    """
    if type(pd_input) == pd.DataFrame:
        pd_input_min = pd_input.min(axis=axis) # input min value
        pd_input_max = pd_input.max(axis=axis) # input max value
        pd_output = lb + (pd_input - pd_input_min)/(pd_input_max - pd_input_min) * (ub - lb) # compute rescaled dataset
        
    elif type(pd_input) == np.ndarray:
        pd_input_min = np.nanmin(pd_input, axis=axis) # input min value
        pd_input_max = np.nanmax(pd_input, axis=axis) # input max value
        pd_output = lb + (pd_input - pd_input_min)/(pd_input_max - pd_input_min) * (ub - lb) # compute rescaled dataset
    
    return pd_output


def concat_nans_1d(input_var, N):
    """
    Concatenate N NaN elements at the end of the input array. This function
    is used as preparation step before input in the 1D CNN
    
    Args:
        input_var = input 1D array (ndarray)
        N = integer with size of new array
    Returns:
        output_var = new 1D array with NaNs concatenated at the end
    """
    input_var_shape = input_var.shape
    if input_var_shape[0] < N:
        if type(input_var) == pd.DataFrame:
            pd_nan = pd.DataFrame(np.zeros(shape=(N-input_var_shape[0], input_var_shape[1]), dtype=np.float32), columns=list(input_var.keys()))*np.nan
            output_var = pd.concat([input_var, pd_nan], axis=0, ignore_index=True)
        
        elif type(input_var) == np.ndarray:
            if len(input_var_shape) == 2:
                np_nan = np.zeros(shape=(N-input_var_shape[0], input_var_shape[1]), dtype=np.float32)*np.nan
                output_var = np.concatenate((input_var, np_nan))
            
            elif len(input_var_shape) == 1:
                np_nan = np.zeros(shape=N-input_var_shape[0], dtype=np.float32)*np.nan
                output_var = np.concatenate((input_var, np_nan))
        
        return output_var
    
    elif input_var_shape[0] == N:
        return input_var
    
    else:
        raise ValueError('Size of input array greater than the size of new array. The size of the new array must be greater than the input array')
        return None

def products_from_npz(common_date, npz_list):
    """
    Takes the common dates between SRAL, SLSTR and OLCI products (folders) and
    finds the same npz files
    
    Args:
        common_date = dictionary that includes the files that are commong among
                      SRAL, SLSTR and OLCI {'SRAL': names of files, 'SLSTR':
                          names of files, 'OLCI': names of files}
        npz_list = list of NPZ files
    Returns:
        output_common_date = dictionary that includes the common dates of the
                             npz files
    """
    output_common_date = {}
        
    if 'SRAL' in common_date.keys():
        # Convert npz_list to datetimes
        npz_sral = [dt.datetime.strptime(item[4:23], '%Y-%m-%d %H_%M_%S') for item in npz_list]
        # Convert common_dates to datetimes
        common_dates_sral = [dt.datetime.strptime(item[16:31], '%Y%m%dT%H%M%S') for item in common_date['SRAL']]
        # Find common npz and products
        output_common_date['SRAL'] = [item_3 for item in npz_sral for (item_2, item_3) in zip(common_dates_sral, common_date['SRAL']) if item == item_2]
    
    if 'SLSTR' in common_date.keys():
        
        npz_slstr = [dt.datetime.strptime(item[25:44], '%Y-%m-%d %H_%M_%S') for item in npz_list]
        common_dates_slstr = [dt.datetime.strptime(item[16:31], '%Y%m%dT%H%M%S') for item in common_date['SLSTR']]
        output_common_date['SLSTR'] = [item_3 for item in npz_slstr for (item_2, item_3) in zip(common_dates_slstr, common_date['SLSTR']) if item == item_2]
    
    if 'OLCI' in common_date.keys():
        npz_olci = [dt.datetime.strptime(item[-23:-4], '%Y-%m-%d %H_%M_%S') for item in npz_list]
        common_dates_olci = [dt.datetime.strptime(item[16:31], '%Y%m%dT%H%M%S') for item in common_date['OLCI']]
        output_common_date['OLCI'] = [item_3 for item in npz_olci for (item_2, item_3) in zip(common_dates_olci, common_date['OLCI']) if item == item_2]
    
    
    return output_common_date
    
if __name__ == '__main__':
#    my_list = ['S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz', 
#               'S3A_2018-05-14 02_04_56__2018-05-14 02_01_39.npz',
#               'S3A_2018-06-11 01_39_10__2018-06-11 01_35_26.npz']
#    
#    out = pick_npz_dates(my_list, '2018-05-05', 10)

    paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\', '\\'),
             'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\SLSTR'.replace('\\','\\')
             }
    # Folder names with the common dates
    common_date = s3utilities.find_common_dates(paths)
    npz_list = os.listdir(r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\npz_files_sral_slstr'.replace('\\','\\'))
    npz_list = [item for item in npz_list if 'npz' in item]