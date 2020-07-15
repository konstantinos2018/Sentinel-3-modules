# -*- coding: utf-8 -*-
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import matplotlib.pyplot as plt
import numpy as np
#import shapefile as shp
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.geometry import mapping

# =============================================================================
# BEGIN
# =============================================================================
# Define constants
inEPSG = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs '
outEPSG = '+proj=utm +zone=23 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '# Gulf Stream

bound = [-3000000, -1000000, 3625000, 4875000] # Gulf Stream
shppath = r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\USA_26923.shp'.replace('\\', '\\') # Gulf Stream 1

# Read shapefile
#polys = shp.Reader(shppath)
polys = gpd.read_file(shppath)
polys = polys.geometry[0]

# Define colors for shapefile
cbrown = '#CD853F'
cblack = '#000000'

# Create figure and take axis handle
fig = plt.figure(figsize=(10,5))
ax = fig.gca()

# Plot shapefile
polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
ax.add_patch(polys_plot)   

grid_step = 330*35 # Distance between points [meters]
x_min = -2750000 # x coordinate min [meters]
x_max = -1250000 # x coordinate max [meters]
y_min = 3800000 # y coordinate min [meters]
y_max = 4700000 # y coordinate max [meters]

n_x = (x_max - x_min)//grid_step # number of points in x axis
n_y = (y_max - y_min)//grid_step # number of points in y axis
x_array = np.linspace(x_min, x_max, n_x) # x coordinates array
y_array = np.linspace(y_min, y_max, n_y) # y coordinates array
index_x = np.arange(n_x) # indices of x coordinates array
index_y = np.arange(n_y) # indices of y coordinates array

x_grid, y_grid = np.meshgrid(x_array, y_array, indexing='ij') # x, y coordinates grid
index_grid_x, index_grid_y = np.meshgrid(index_x, index_y, indexing='ij') # x, y indices grid
x_array = np.matrix.flatten(x_grid) # x coordinates array of grid
y_array = np.matrix.flatten(y_grid) # y coordinates array of grid
index_x = np.matrix.flatten(index_grid_x) # indices of x coordinates array of grid
index_y = np.matrix.flatten(index_grid_y) # indices of y coordinates array of grid

ax.scatter(x_array, y_array, color='r')
ax.set_xlim(-3000000, -1100000)
ax.set_ylim(3700000, 4800000)
# Points to GeoSeries
pts = gpd.GeoSeries([Point(x, y) for x, y in zip(x_array, y_array)], crs={'init':'epsg:26923'})
pts.to_file(r'D:\vlachos\Documents\KV MSc thesis\Data\Country_borders\grid_true_GulfStream.shp'.replace('\\','\\'))
#pts_outside = ~pts.within(polys['geometry'].loc[0])

