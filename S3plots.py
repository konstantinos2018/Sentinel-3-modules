# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:14:34 2018

@Description: The module generates and saves differents kinds of graphs
@author: Kostas Vlachos
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import pdb
import matplotlib.colors as colors
import scipy.stats as scst
import geopandas as gpd
# My Modules
import nc_manipul as ncman
import S3coordtran as s3ct


def sral_scatter(lon, lat, ssha, shppath, bound, fdate, plotpath):
    """
    Plots and saves SRAL on a map in the form of a  scatterplot.
    
    Args:
        lon = 1d array of SRAL x coordinates
        lat = 1d array of SRAL y coordinates
        ssha = 1d array of SRAL variable
        bound = [xmin, xmax, ymin, ymax]
        fdate = string of the date of the file which will be used on the graph title
                as well as the name of the PNG file
        plotpath = string with the path of the plot
    Returns:
        None
    """
    
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
    # Create figure and take axis handle
    fig = plt.figure()
    ax = fig.gca()
    
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    # colorbar colormap
#    cm = plt.cm.get_cmap('cool_r')
    
    # Plot variable
#    sc = plt.scatter(lon, lat, c=ssha, marker='.', s=10**2, cmap=cm,
#                     vmin=-0.2, vmax=0.5)
    
    # segments
    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#    fig, axs = plt.subplots(1,1,sharex=True,sharey=True)
    norm = plt.Normalize(ssha.min(), ssha.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    
    lc.set_alpha(1)
    lc.set_array(ssha)
    lc.set_linewidth(8)
    line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax)
    
    # colorbar, title, label
#    cbar = plt.colorbar(sc)
    plt.title('SRAL ' + fdate, fontdict={'fontsize': 23})
    cbar.ax.set_ylabel('SSHA [m]', rotation=270, labelpad=20, fontsize=18)
    
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    
    # Labels
    plt.xlabel('X [m]', fontsize=18)
    plt.ylabel('Y [m]', fontsize=18)
    
    # Rotate axis labels
    plt.xticks(rotation=45)
    
    # Set figure size
    fig.set_size_inches(18, 10)

    # Save plot
    plt.savefig(plotpath + '\\SRAL ' + fdate + '.png',
                dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None


def ja3_scatter(lon, lat, ssh, shppath, bound):
    """
    Plot Jason-3 variables
    
    Input:
        lon = 1d array of longitude/x
        lat = 1d array of latitude/y
        ssha = 1d array of some variable (e.g. ssha etc.)
    Output:
        None
    """
    
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
    # Create figure and take axis handle
    fig = plt.figure()
    ax = fig.gca()
    
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    
    # colorbar colormap
    cm = plt.cm.get_cmap('RdYlBu_r')
    
    # Plot variable
    sc = plt.scatter(lon, lat, c=ssh, marker='.', s=10**2, cmap=cm)
                     #vmin=-0.2, vmax=0.5)
    
    # colorbar, title, label
    cbar = plt.colorbar(sc)
    plt.title('Jason-3 Altimetry')
    cbar.ax.set_ylabel('SSH [m]', rotation=270, labelpad=20)
    
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    
    # Labels
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    
    # Rotate axis labels
    plt.xticks(rotation=45)
    
    # Set figure size
    fig.set_size_inches(10, 10)
    
    return None


def olci_scatter(lon, lat, var, shppath, bound, fdate, plotpath):
    """
    Plots and saves OLCI variables (or even SLSTR) on a map in the form of
    a scatterplot
    
    Args:
        lon = 1d array of OLCI x coordinates
        lat = 1d array of OLCI y coordinates
        var = 1d array of OLCI variable
        shppath = string with the file of the shapefile of the region
        bound = [xmin, xmax, ymin, ymax]
        fdate = string with the titl and name of file of the graph
        plotpath = string with the path of the plot
    Returns:
        None
    """
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
    font = {'size' : 18}
    plt.rc('font', **font)
    
    # Create figure and take axis handle
    fig = plt.figure()
    ax = fig.gca()
    
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    
    # colorbar colormap
    cm = plt.cm.get_cmap('viridis')
#    cm = plt.reverse_colourmap(cm)
    
    if scst.skew(var) > 0:
        vmin = np.nanmin(var)
        vmax = np.median(var) + 2.5*np.std(var)
    else:
        vmin = np.median(var) - 2.5*np.std(var)
        vmax = np.nanmax(var)

    # Plot variable
    sc = plt.scatter(lon, lat, c=var, marker='.', s=1**2, cmap=cm,
                     vmin=vmin, vmax=vmax) #vmin=-1.8, vmax=1.5)
    
    # colorbar, title, label
    cbar = plt.colorbar(sc)
    plt.title(fdate, fontdict={'fontsize': 23})
    cbar.ax.set_ylabel('Chlorophyll log(mg/m^3)', rotation=270, labelpad=20, fontsize=18)
    
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    
    # Labels
    plt.xlabel('X [m]', fontsize=18)
    plt.ylabel('Y [m]', fontsize=18)
    
    # Rotate axis labels
    plt.xticks(rotation=45)
    
    # Set figure size
    fig.set_size_inches(18, 10)
    
    # Save plot
#    plt.savefig('D:\\vlachos\\Desktop\\olci.png', dpi=300)
    
    # Save plot
    plt.savefig(os.path.join(plotpath,'OLCI ' + fdate + '.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None

def slstr_scatter(lon, lat, var, shppath, bound, fdate, plotpath):
    """
    Plots and saves SLSTR on a map in the form of a  scatterplot.
    
    Args:
        lon = 1D array of SLSTR x coordinates
        lat = 1D array of SLSTR y coordinates
        var = 1D array of SLSTR variable
        shppath = string with the file of the shapefile of the region
        bound = [xmin, xmax, ymin, ymax]
        fdate = dictionary with info about the graph {'plttitle': plot title,
                                                      'fiilename': name of file'}
        plotpath = string with the path of the plot
    Returns:
        None
    """
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
#    # Compute ration for the output figure
#    fig_ratio = np.abs((bound[1] - bound[0])/(bound[2] - bound[3]))
#    if fig_ratio > 1:
#        fig_size_tuple = (10, 10*1.0/fig_ratio + 3)
#    else:
#        fig_size_tuple = (10*1.0/fig_ratio + 3, 10)
    # Create figure and take axis handle
    fig = plt.figure(figsize=(18, 10))
    ax = fig.gca()
    
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot) 
    
    # colorbar colormap
    cm = plt.cm.get_cmap('RdYlBu_r')
#    cm = plt.reverse_colourmap(cm)
    
    # Take certain percentiles of SST in order to exclude extreme values and therefore
    # have better color scale visualization
    idx = (var > np.percentile(var, 3)) & (var < np.percentile(var, 97))
    sst_temp = var[idx]
    
    # Check if sst_temp is empty after indexing
    if sst_temp.size == 0:
        return None
    
    # Define minimum value for the SST colourbar
    if sst_temp.min() < 0:
        sstmin = 0
        sstmax = sst_temp.max()
    else:
        sstmin = sst_temp.min()
        sstmax = sst_temp.max()
        
    # Plot variable
    sc = plt.scatter(lon, lat, c=var, marker='.', s=1**2, cmap=cm,
                     vmin=sstmin, vmax=sstmax)
    
    # colorbar, title, label
    cbar = plt.colorbar(sc)
    plt.title(fdate, fontdict={'fontsize': 23})
    cbar.ax.set_ylabel('SST [$^\circ$C]', rotation=270, labelpad=20, fontsize=18)
    
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    
    # Labels
    plt.xlabel('X [m]', fontsize=18)
    plt.ylabel('Y [m]', fontsize=18)
    
    # Rotate axis labels
    plt.xticks(rotation=45)
    
    # Set figure size
#    fig.set_size_inches(10, 10)
    
    # Save plot
#    plt.savefig('D:\\vlachos\\Desktop\\sst.png', dpi=300)
    
    # Save plot
    plt.savefig(plotpath + '\\SLSTR ' + fdate + '.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None

def slstr_sral_trajplot(xsr, ysr, ssh, xsl, ysl, sst, shppath, bound, fdate, plotpath):
    """
    Plots and saves the input variables (i.e. SRAL and SLSTR) on a map in the
    form of a  scatterplot.
    
    Args:
        xsr = 1D array of SRAL x coordinates
        ysr = 1D array of SRAL y coordinates
        ssh = 1D array of SRAL variable
        xol = 1D array of SLSTR x coordinates
        yol = 1D array of SLSTR y coordinates
        chl = 1D array of SLSTR variable
        shppath = string with the file of the shapefile of the region
        bound = [xmin, xmax, ymin, ymax]
        fdate = dictionary with info about the graph {'plttitle': plot title,
                                                      'fiilename': name of file'}
        plotpath = string with the path of the plot
    Returns:
        None
    """
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
    # Create figure and take axis handle
    fig = plt.figure(figsize=(18,10))
    ax = fig.gca()
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))   
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    
    # colorbar colormap
    cm = plt.cm.get_cmap('RdYlBu_r')
    #    cm = plt.reverse_colourmap(cm)
    
    # Take certain percentiles of SST in order to exclude extreme values and therefore
    # have better color scale visualization
    idx = (sst > np.percentile(sst, 3)) & (sst < np.percentile(sst, 97))
    sst_temp = sst[idx]
    if sst_temp.size == 0:
        return None
    # Define minimum value for the SST colourbar
    if sst_temp.min() < 0:
        sstmin = 0
        sstmax = sst_temp.max()
    else:
        sstmin = sst_temp.min()
        sstmax = sst_temp.max()
    # Define minimum and maximum value for the SRAL colourbar
#    idx = (ssh > np.percentile(ssh, 15)) & (ssh < np.percentile(ssh, 85))
#    sshmin = ssh[idx].min()
#    sshmax = ssh[idx].max()
    
    # Plot variable
    sc = plt.scatter(xsl, ysl, c=sst, marker='.', s=1**2, cmap=cm,
                     vmin=sstmin, vmax=sstmax)
    # colorbar, title, label
    cbar = plt.colorbar(sc)
    plt.title(fdate['plttitle'], fontdict={'fontsize': 20})
    cbar.ax.set_ylabel('SST [$^\circ$C]', rotation=270, labelpad=20, fontsize=18)
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    # Labels
    plt.xlabel('X [m]', fontsize=18)
    plt.ylabel('Y [m]', fontsize=18)
    # Rotate axis labels
    plt.xticks(rotation=45)
    # Set figure size
    
    # segments
    points = np.array([xsr, ysr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#    fig, axs = plt.subplots(1,1,sharex=True,sharey=True)
    norm = plt.Normalize(ssh.min(), ssh.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    
    lc.set_alpha(1)
    lc.set_array(ssh)
    lc.set_linewidth(8)
    line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax)
    
    
#    # colorbar colormap
#    cm = plt.cm.get_cmap('cool_r')
#    
#    # Plot variable
#    sc = plt.scatter(xsr, ysr, c=ssh, marker='.', s=10**2, cmap=cm,
#                     vmin=-0.2, vmax=0.5)
        
    # colorbar, title, label
#    cbaxes = fig.add_axes([-0.05, 0.1, 0.035, 0.75])
#    cbar = plt.colorbar(sc)#, cax=cbaxes)
    cbar.ax.set_ylabel('SSHA [m]', rotation=270, labelpad=10, fontsize=18)#, labelpad=-60)
#    cbar.ax.position([5,10, 2, 5])
    # Save plot
    plt.savefig(plotpath + '\\' + fdate['filename'] + '.png',
                dpi=300, bbox_incehs='tight')
    
    plt.close('all')
    
    return None

def olci_sral_trajplot(xsr, ysr, ssh, xol, yol, chl, shppath, bound, fdate):
    """
    Plots and saves the input variables (i.e. SRAL and OLCI) on a map in the
    form of a  scatterplot.
    
    Args:
        xsr = 1D array of SRAL x coordinates
        ysr = 1D array of SRAL y coordinates
        ssh = 1D array of SRAL variable
        xol = 1D array of OLCI x coordinates
        yol = 1D array of OLCI y coordinates
        chl = 1D array of OLCI variable
        shppath = string with the file of the shapefile of the region
        bound = [xmin, xmax, ymin, ymax]
        fdate = dictionary with info about the graph {'plttitle': plot title,
                                                      'fiilename': name of file'}
    Returns:
        None
    """
    # Read shapefile
#    polys = shp.Reader(shppath)
    polys = gpd.read_file(shppath)
    polys = polys.geometry[0]
    
    # Define colors for shapefile
    cbrown = '#CD853F'
    cblack = '#000000'
    
    # Create figure and take axis handle
    fig = plt.figure(figsize=(18,10))
    ax = fig.gca()
    
    # Plot shapefile
#    for i in polys.shapes():
#        poly = i.__geo_interface__
#        ax.add_patch(PolygonPatch(poly, fc=cbrown, ec=cblack, alpha=0.5,zorder=2))   
    polys_plot = PolygonPatch(polys, fc=cbrown, ec=cblack, alpha=0.5,zorder=2)
    ax.add_patch(polys_plot)
    
    # colorbar colormap
    cm = plt.cm.get_cmap('viridis')
    #    cm = plt.reverse_colourmap(cm)
    
    if scst.skew(chl) > 0:
        vmin = np.nanmin(chl)
        vmax = np.median(chl) + 2.5*np.std(chl)
    else:
        vmin = np.median(chl) - 2.5*np.std(chl)
        vmax = np.nanmax(chl)
    
    # Plot variable
    sc = plt.scatter(xol, yol, c=chl, marker='.', s=1**2, cmap=cm,
                     vmin=vmin, vmax=vmax)
    # colorbar, title, label
    cbar = plt.colorbar(sc)
    plt.title(fdate['plttitle'], fontdict={'fontsize': 23})
    cbar.ax.set_ylabel('Chlorophyll log(mg/m^3)', rotation=270, labelpad=20, fontsize=18)
    # Set axis limits
    plt.xlim(bound[0],bound[1])
    plt.ylim(bound[2],bound[3])
    # Labels
    plt.xlabel('X [m]', fontsize=18)
    plt.ylabel('Y [m]', fontsize=18)
    # Rotate axis labels
    plt.xticks(rotation=45)
    
    # segments
    points = np.array([xsr, ysr]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
#    fig, axs = plt.subplots(1,1,sharex=True,sharey=True)
    norm = plt.Normalize(ssh.min(), ssh.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    
    lc.set_alpha(1)
    lc.set_array(ssh)
    lc.set_linewidth(8)
    line = ax.add_collection(lc)
    cbar = plt.colorbar(line, ax=ax)
    
    # Set figure size
#    fig.set_size_inches(10, 10)
    
    #fig = plt.figure()
    # colorbar colormap
#    cm = plt.cm.get_cmap('cool_r')
    
    # Plot variable
#    sc = plt.scatter(xsr, ysr, c=ssh, marker='.', s=10**2, cmap=cm,
#                     vmin=-0.2, vmax=0.5)
        
    # colorbar, title, label
#    cbaxes = fig.add_axes([-0.05, 0.1, 0.035, 0.75])
#    cbar = plt.colorbar(sc)#, cax=cbaxes)
    cbar.ax.set_ylabel('SSHA [m]', rotation=270, fontsize=18)#, labelpad=-10)
    
    # Save plot
    plt.savefig(r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Mediterranean_Test\SRAL_OLCI'.replace('\\','\\') + '\\' + fdate['filename'] + '.png', 
                dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None


def scatter_sral_slstr(sral, slstr, fname, axis_labels, plotpath):
    """
    Plots and saves the input variables (i.e. SRAL and SLSTR) that are along the
    track in the form of a  scatterplot.
    
    Args:
        sral = 1D array of SRAL variable
        slstr = 1D array of SLSTR variable
        fname = dictionary with information about the graph {'plttitle':plot title,
                                                             'filename': name of file}
        axis_labels = dictionary with the labels of each axis {'X': xlabel,
                                                               'Y': ylabel}
        plotpath = string with the path of the plot
    Returns:
        None
    """

    font = {'size' : 18}
    plt.rc('font', **font)
    
    # Create figure and take axis handle
    plt.figure(figsize=(18,10))
#    ax = fig.gca()
    
    plt.title(fname['plttitle'], fontdict={'fontsize': 23})
    
    # Plot variable
    plt.scatter(slstr, sral, marker='.', s=10**2)
    
    plt.xlabel(axis_labels['X'], fontsize=18)
    plt.ylabel(axis_labels['Y'], fontsize=18)
    
    # Save plot
    plt.savefig(plotpath + '\\scat_' + fname['filename'] + '.png', 
                dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None

def histogram_sral(sral, fname, axis_labels):
    """
    Plots a histogram of the SRAL variable values. It can also be used for
    other variables
    
    Args:
        sral = ndarray with the SRAL variable values
        fname = dictionary with info about the graph {'plttitle': title of plot,
                                                      'filename': name of file}
        axis_labels = dictionary with info about the graph axis {'X': name of
                                                                 x label,
                                                                 'Y': name of y                                                               label}
    Returns:
        None
    """
    # Create figure and take axis handle
    plt.figure(figsize=(18,10))
#    ax = fig.gca()
    
    plt.title(fname['plttitle'], fontdict={'fontsize': 23})
    
    # Plot variable
    plt.hist(sral, bins='fd')
    
    plt.xlabel(axis_labels['X'], fontsize=18)
    plt.ylabel(axis_labels['Y'], fontsize=18)
    
    # Save plot
    plt.savefig(r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\SRAL_clean\Histogram'.replace('\\','\\') + '\\hist_' + fname['filename'] + '.png', 
                dpi=300, bbox_inches='tight')
    
    plt.close('all')
    
    return None


def sral_cross_sections(variables, distance, fname, plotpath):
    """
    Plots SRAL/SLStr against Distance of a specific track
    
    Args:
        variables = dictionary with variables (keys=variable name,
                                               values=1D ndarray)
        distance = dictionary with 1D arrays distances {'SRAL': sral variable,
                                                        'SLSTR': slstr variable}
        fname = dictionary with information about the graph {'plttitle':plot title,
                                                             'filename': name of file}
        plotpath = string with the path of the plot
    Returns:
        None
    """
    # Assign to variables
    ssh = variables['SRAL']
    sst = variables['SLSTR']
    
    dst_ssh = distance['SRAL']
    dst_sst = distance['SLSTR']
    
    font = {'size' : 18}
    plt.rc('font', **font)

    # Plot
    fig = plt.figure(figsize=(18,10))
    
    plt.title(fname['plttitle'], fontdict={'fontsize': 23})
    
    plt.plot(dst_ssh, ssh, 'b-', alpha=0.6, linewidth=1)
    plt.plot(dst_sst, sst, 'r-', linewidth=1)
    
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('Variable', fontsize=18)
    
#    plt.legend(('SSHA [m]', 'SST [$^\circ$C]'), fontsize=23)
    plt.legend(('SSHA [m]', 'chl [log(mg/$m^3$)]'), fontsize=23)
    
    # Save plot
    plt.savefig(plotpath + '\\traj_' + fname['filename'] + '.png', 
                dpi=300, bbox_inches='tight')

    plt.close('all')
    
    return None

def sral_cross_sections_olci(variables, distance, fname, plotpath):
    """
    Plots and saves the cross-section plot between SRAL/OLCI variables
    and distance of a track.
    
    Args:
        variables = dictionary with variables (keys=variable name,
                                               values=1D ndarray)
        distance = dictionary with 1D arrays distances {'SRAL': sral variable,
                                                        'OLCI': slstr variable}
        fname = dictionary with information about the graph {'plttitle':plot title,
        'filename': name of file}
        plotpath = string with the path of the plot
    Returns:
        None
    """
    
    # Assign to variables
    ssh = variables['SRAL']
    olci = variables['OLCI']
    
    dst_ssh = distance['SRAL']
    dst_olci = distance['OLCI']
    
    font = {'size' : 18}
    plt.rc('font', **font)

    # Plot
    fig = plt.figure(figsize=(18,10))
    
    plt.title(fname['plttitle'], fontdict={'fontsize': 23})
    
    plt.plot(dst_ssh, ssh, 'b-', alpha=0.6, linewidth=1)
    plt.plot(dst_olci, olci, 'y-', linewidth=1.6)
    
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('Variable', fontsize=18)
    
#    plt.legend(('SSHA [m]', 'CHL_OC4ME [log(mg/$m^3$)]'), fontsize=23)
#    plt.legend(('SSHA [m]', 'TSM_NN [1/m)]'), fontsize=23)
    plt.legend(('SSHA [m]', 'ADG443_NN [$m^{-1}$]'), fontsize=23)
    # Save plot
    plt.savefig(plotpath + '\\traj_' + fname['filename'] + '.png', 
                dpi=300, bbox_inches='tight')

    plt.close('all')
    
    return None


def multiple_cross_sections(var_in, distance, legend_labels, fname, plotpath):
    """
    Plots and saves multiple variables in a cross-section from a track
    
    Args:
        var_in = dictionary with variables names and values (keys=variables names, 
                 values=list with variable values as numpy arrays and kwargs for the plot) (y axis)
                 E.g. var_in = {'SSHA Moving Average': [numpy array, dictionary (kwargs plot)],
                                'SSHA smoothing': [numpy array, dictionary (kwargs plot)]
                                }
        distance = 1D array with distance (x axis)
        legend_labels = tuple of strings (e.g. ('SSHA smooth', 'SSHA moving average'))
        fname = string of the filename and title
    Returns:
        None
    """

    font = {'size' : 18}
    plt.rc('font', **font)
    
    # Plot
    fig = plt.figure(figsize=(18,10))
    
    plt.title(fname, fontdict={'fontsize': 23})

    plt.rc('grid', linestyle='--', color='black')
    plt.grid(True, axis='x')
        
    for name in legend_labels:
        plt.plot(distance, var_in[name][0], **var_in[name][1])
    
    plt.legend(legend_labels, fontsize=23)

        
    plt.xlabel('Distance [m]', fontsize=18)
    plt.ylabel('SSHA [m]', fontsize=18)
#    plt.ylabel('SST [$^\circ$C]', fontsize=18)
         
    plt.savefig(plotpath + '\\' + fname + '.png', 
                dpi=300, bbox_incehs='tight')

    plt.close('all')
    
    return None

def cross_correl_2variable(var_in, num_lag, dicinp, plotpath):
    """
    Plots and saves the linear global cross-correlation of a track using
    matplotlib.pyplot.xcorr()
    
    Args:
        var_in = dictionary with variables (keys=variable name, values=1D ndarray)
        num_lag = positive integer with the maximum number of lags
        dicinp = dictionary with info about the graph {'plttile': plot title,
           'filename':name of file}
        plotpath = string with the path of the plot
    Returns:
        None
    """
    # Assign to variables
    ssh = var_in['SRAL']
    slstr = var_in['SLSTR']
    
    font = {'size' : 18}
    plt.rc('font', **font)

    # Cross corel
    fig = plt.figure(figsize=(18, 10))
    # cross-correlation x1 vs x2
    plt.xcorr(ssh, slstr, usevlines=True, maxlags=num_lag, normed=True, lw=2)
    plt.title(dicinp['plttitle'], fontsize=23)
    plt.xlabel('Lag [# elements]', fontsize=18)
    plt.ylabel('Cross-Corr', fontsize=18)
    plt.ylim([-1, 1])
    plt.xlim([-num_lag, num_lag])
    plt.grid(True)
    
    plt.savefig(plotpath + '\\' + dicinp['filename'] + '.png', 
            dpi=300, bbox_inches='tight')
        
    plt.close('all')
    
    return None

if __name__ == '__main__':
#    inEPSG = 'epsg:4326'
#    outEPSG = 'epsg:3035'
    
    None