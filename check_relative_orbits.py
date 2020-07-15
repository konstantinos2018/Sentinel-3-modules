# -*- coding: utf-8 -*-
""" Read the XML metadata files of Sentinel-3 products of each date and records
    the relative orbit number. The complete record of orbit numbers is plotted
    in a histogram form. The product type (i.e. SRAL, SLSTR or OLCI) is defined
    via the given path
    
    ***IMPORTANT*** The use of the function s3utilities.read_xml_S3_relative_orbits()
    is controversial. Check it.
"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
# My Modules
import s3utilities

# =============================================================================
# BEGIN
# =============================================================================
path_sr = r'H:\MSc_Thesis_05082019\Data\Satellite\Gulf Stream_1\SRAL'.replace('\\','\\') # path of SRAL file

xml_name = 'xfdumanifest.xml' # metadata filename

# List product folders
folder_sral = os.listdir(path_sr)
folder_sral = [item for item in folder_sral if 'S3A' in item or 'S3B' in item]

rel_orbs = [] # relative orbits
for folder in folder_sral:
    # path of metadata file
    fullpath = os.path.join(os.path.join(path_sr, folder), xml_name)

    # Find and list relative orbit of file
    relative_orbit = s3utilities.read_xml_S3_relative_orbits(fullpath)
#    rel_orbs.append(relative_orbit)
    rel_orbs.extend(relative_orbit)

#pdb.set_trace()
# Convert to ndarray
rel_orbs = [int(item) for item in rel_orbs]
rel_orbs = np.array(rel_orbs)
# keep unique which will be the number of bins of histogram
n_bins = np.unique(rel_orbs).size

# =============================================================================
# PLOT
# =============================================================================
font = {'size' : 18}
plt.rc('font', **font)
plt.figure(figsize=(18, 8))
plt.hist(rel_orbs, bins=n_bins, color='orange')
plt.xlabel('# of Relative Orbit', fontsize=18)
plt.ylabel('# of counts', fontsize=18)
plt.title('Relative Orbits over study area', fontsize=23)
plt.xticks(rotation=45)
#fig = plt.figure(figsize=(12,8))
#plt.title(filename[:-4], fontsize=23)
#plt.ylabel('%', fontsize=18)
#plt.bar(range(len(matrix_labels)), importances*100, color='b', align="center")
#plt.xticks(range(len(matrix_labels)), matrix_labels, rotation = 45)
