# -*- coding: utf-8 -*-
# =============================================================================
# DESCRIPTION
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.colors as colors
import datetime as dt
# My Modules
import ml_utilities

# =============================================================================
# BEGIN
# =============================================================================
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
# =============================================================================
# GRID WITH QUERY POINTS
# =============================================================================
#spatial_resolution = 330 # spatial resolution/pixel size of altimetry [meters]
#sral_winsize = 35 # the window size that was used during the SSHA filtering
#                  # in order to produce the features. 35 corresponds to ~11.55 km
#                  # This will be the spatial resolution and pixel size of the grid
#grid_step = spatial_resolution*sral_winsize # Distance between points [meters]
#x_min = -2750000 # x coordinate min [meters]
#x_max = -1250000 # x coordinate max [meters]
#y_min = 3800000 # y coordinate min [meters]
#y_max = 4700000 # y coordinate max [meters]
#
#x_query, y_query, n_x, n_y = ml_utilities.create_grid_true(x_min, x_max, y_min, y_max, grid_step)
#
#del grid_step, spatial_resolution, sral_winsize, x_min, x_max, y_min, y_max

# =============================================================================
# BUILD FEATURE MATRIX ON THE QUERY POINTS
# =============================================================================

# =============================================================================
# LOAD RANDOM FOREST MODEL
# =============================================================================
path_models = r'C:\Users\vlachos\Desktop\grid_npz_sral_slstr_olci_perTrack_RF\models'.replace('\\','\\')
#filename = 'RF_complete_data_model.sav'
#model = pickle.load(open(os.path.join(path, filename), 'rb'))

path_grid = r'C:\Users\vlachos\Desktop\grid_npz_sral_slstr_olci_perTrack_RF'.replace('\\','\\')
#filename = 'S3A_2018-05-10 02_08_39__2018-05-10 02_05_24_grid.npz'
#X, Y, matrix, other_variables = ml_utilities.grid_feature_matrix_from_npz(os.path.join(path, filename))
models_list = os.listdir(path_models)
models_list = [item for item in models_list if '.sav' in item]
npz_list = os.listdir(path_grid)
npz_list = [item for item in npz_list if '.npz' in item]

empty_frames = []

for model_name in models_list:
    sral_model = dt.datetime.strptime(model_name[4:23], '%Y-%m-%d %H_%M_%S')
    slstr_model = dt.datetime.strptime(model_name[25:44], '%Y-%m-%d %H_%M_%S')
    olci_model = dt.datetime.strptime(model_name[-43:-24], '%Y-%m-%d %H_%M_%S')
    
    for npz_file_grid in npz_list:
        f_sral = dt.datetime.strptime(npz_file_grid[4:23], '%Y-%m-%d %H_%M_%S')
        f_slstr = dt.datetime.strptime(npz_file_grid[25:44], '%Y-%m-%d %H_%M_%S')
        f_olci = dt.datetime.strptime(npz_file_grid[46:65], '%Y-%m-%d %H_%M_%S')
        
        if (f_sral == sral_model) and (f_slstr == slstr_model) and (f_olci == olci_model):
            pass
        else:
            continue

        # Read model
        model = pickle.load(open(os.path.join(path_models, model_name), 'rb'))
        # Read grid and features
        X, Y, matrix, other_variables = ml_utilities.grid_feature_matrix_from_npz(os.path.join(path_grid, npz_file_grid))
        n_x = other_variables['n_x']
        n_y = other_variables['n_y']
        
        # Re-order features so they will be the same order as when the models where trained
        matrix = matrix.reindex(columns=['KD490_M07_OLCI_150km',
               'SST_32km','SST_125km','TSM_NN_OLCI_95km',
               'SST_95km','KD490_M07_OLCI_32km','ADG443_NN_OLCI_150km',
               'TSM_NN_OLCI_32km','CHL_OC4ME_OLCI_95km',
               'SST_53km','SST_12.5km','KD490_M07_OLCI_95km',
               'ADG443_NN_OLCI_95km','CHL_OC4ME_OLCI_150km',
               'SST_150km','CHL_OC4ME_OLCI_32km',
               'ADG443_NN_OLCI_32km','TSM_NN_OLCI_150km'])
    
        
        # =============================================================================
        # APPLY MODEL
        # =============================================================================
        matrix_2, idx_nan = ml_utilities.imputate_nans_feature_matrix(matrix, method='Kickout')
        if matrix_2.empty:
            empty_frames.append(npz_file_grid)
            continue
        
        y_hat = model.predict(matrix_2)
        
        # Recreate grid
        y_hat_new = np.zeros(shape=matrix.shape[0])*np.nan
        y_hat_new[~idx_nan] = y_hat
        
        y_hat_new = y_hat_new.reshape([n_x, n_y])
        X = X.reshape([n_x, n_y])
        Y = Y.reshape([n_x, n_y])
        
        # =============================================================================
        # PLOT
        # =============================================================================
        shppath = r'C:\Users\vlachos\Desktop\grid_npz_sral_slstr_olci_perTrack_RF\shp\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1
        
        # Read shapefile
        #polys = shp.Reader(shppath)
        polys = gpd.read_file(shppath)
        # Define colors for shapefile
        cbrown = '#CD853F'
        cblack = '#000000'
        
        # Create figure and take axis handle
        #fig = plt.figure(figsize=(10,5))
        font = {'size' : 18}
        plt.rc('font', **font)
        fig = plt.figure(figsize=(15,10))
        ax = fig.gca()
        polys_plot = polys.geometry[0]
        polys_plot = PolygonPatch(polys_plot, fc=cbrown, ec=cblack, alpha=1,zorder=1)
        ax.add_patch(polys_plot) 
            
        cm = plt.cm.get_cmap('jet')
#        norm = MidpointNormalize(midpoint=0)
        norm = plt.Normalize(-1,1)
        #sc = plt.scatter(X[matrix_2.index], Y[matrix_2.index], c=y_hat, cmap=cm, marker='.', s=10**1.8)
#        vmin = np.nanmean(y_hat_new.flatten()) - 2.5*np.nanstd(y_hat_new.flatten())
#        vmax = np.nanmean(y_hat_new.flatten()) + 2.5*np.nanstd(y_hat_new.flatten())
        sc = ax.pcolor(X,Y, y_hat_new, cmap=cm, norm=norm)#, vmin=vmin, vmax=vmax)#, norm=norm)
        cbar = plt.colorbar(sc)
        plt.title('{0}\nSRAL {1}\nSLSTR {2}\nOLCI {3}'.format(model_name[:3], model_name[4:23], model_name[25:44], model_name[-43:-24]), fontsize=23)
        #plt.clim(0.035,0.065)
        cbar.ax.set_ylabel('SSHA [m]', rotation=270, labelpad=20, fontsize=18)
        plt.xlim(-3000000, -1000000)
        plt.ylim(3625000, 4875000)
        plt.xlabel('X [m]', fontsize=18)
        plt.ylabel('Y [m]', fontsize=18)
        plt.xticks(rotation=45)
        
        plotpath = r'C:\Users\vlachos\Desktop\grid_npz_sral_slstr_olci_perTrack_RF\plots'.replace('\\','\\')
        fig.savefig(os.path.join(plotpath, npz_file_grid) + '_SSHAprediction.png', dpi=300, bbox_inches='tight')
        plt.close('all')
#        raise Exception()
    


