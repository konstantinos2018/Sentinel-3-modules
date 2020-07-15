# -*- coding: utf-8 -*-
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
# My Modules
import ml_utilities

# =============================================================================
# GRID WITH QUERY POINTS
# =============================================================================
spatial_resolution = 330 # spatial resolution/pixel size of altimetry [meters]
sral_winsize = 35 # the window size that was used during the SSHA filtering
                  # in order to produce the features. 35 corresponds to ~11.55 km
                  # This will be the spatial resolution and pixel size of the grid
grid_step = 330*35 # Distance between points [meters]
x_min = -2250000 # x coordinate min [meters]
x_max = -1250000 # x coordinate max [meters]
y_min = 4100000 # y coordinate min [meters]
y_max = 4600000 # y coordinate max [meters]

x_query, y_query, n_x, n_y = ml_utilities.create_grid_true(x_min, x_max, y_min, y_max, grid_step)

del grid_step, spatial_resolution, sral_winsize, x_min, x_max, y_min, y_max

# =============================================================================
# BUILD FEATURE MATRIX ON THE QUERY POINTS
# =============================================================================

# =============================================================================
# LOAD RANDOM FOREST MODEL
# =============================================================================
path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\Random_Forest\RF_complete_sral_slstr_olci'.replace('\\','\\')
filename = 'RF_complete_data_model.sav'
model = pickle.load(open(os.path.join(path, filename), 'rb'))

path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\grid_npz_sral_slstr_olci_RF\RF_complete_dataset_model_sral_slstr_olci'.replace('\\','\\')
filename = 'S3A_2018-08-07 01_57_48__2018-08-07 14_48_55.npz'
X, Y, matrix, other_variables = ml_utilities.grid_feature_matrix_from_npz(os.path.join(path, filename))

# Rescale
ub = 1
lb = -1
matrix = ml_utilities.matrix_min_max_rescale(matrix, ub=ub, lb=lb, axis=0)

# =============================================================================
# APPLY MODEL
# =============================================================================
matrix_2, idx_nan = ml_utilities.imputate_nans_feature_matrix(matrix, method='Kickout')
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
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1

# Read shapefile
#polys = shp.Reader(shppath)
polys = gpd.read_file(shppath)
# Define colors for shapefile
cbrown = '#CD853F'
cblack = '#000000'

# Create figure and take axis handle
#fig = plt.figure(figsize=(10,5))
fig = plt.figure(figsize=(15,10))
ax = fig.gca()
polys_plot = polys.geometry[0]
polys_plot = PolygonPatch(polys_plot, fc=cbrown, ec=cblack, alpha=1,zorder=0)
ax.add_patch(polys_plot) 
    
font = {'size' : 18}
plt.rc('font', **font)
cm = plt.cm.get_cmap('jet')
#sc = plt.scatter(X[matrix_2.index], Y[matrix_2.index], c=y_hat, cmap=cm, marker='.', s=10**1.8)
sc = plt.pcolor(X,Y, y_hat_new, cmap=cm)
cbar = plt.colorbar(sc)
plt.title('2018-08-07 Sentinel-3 Altimetry', fontsize=23)
#plt.clim(0.035,0.065)
cbar.ax.set_ylabel('SSHA [m]', rotation=270, labelpad=20)
plt.xlim(-3000000, -1000000)
plt.ylim(3625000, 4875000)
plt.xlabel('X [m]', fontsize=18)
plt.ylabel('Y [m]', fontsize=18)
plt.xticks(rotation=45)
    
