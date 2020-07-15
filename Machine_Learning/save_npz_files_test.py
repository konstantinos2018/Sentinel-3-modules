# -*- coding: utf-8 -*-
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np

#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\date1.npz'.replace('\\','\\')
filename = r'H:\MSc_Thesis_14032019\Data\Satellite\date1.npz'.replace('\\','\\')

# Create dummy data 1
ssha = np.array([1,2,3,np.nan,5,6],ndmin=2).reshape([-1,1])
idx = np.isnan(ssha)
ssha = np.ma.array(ssha, mask=idx)

# Create dummy data 2
dst = np.ma.array([10, 20, 30, 40, 50, 60], ndmin=2).reshape([-1,1])

data = {'SSHA': ssha, 'Distance':dst}
# Save file
#np.savez(filename, {'SSHA': ssha, 'Distance':dst})
np.savez_compressed(filename, data)
#np.savez(filename, SSHA=ssha)

dat = np.load(filename)

# Retrieve dictionary
dat = dat['arr_0'].item()
matrix = np.ma.zeros((dat['SSHA'].size, len(dat)), fill_value=-9999)

for i, key in zip(range(len(dat)), dat.keys()):
    if type(dat[key]) is np.ndarray: # Check if value of dictionary is np array
        matrix[:, i] = np.squeeze(dat[key])
    
matrix = np.ma.fix_invalid(matrix)

#%% 
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import matplotlib.pyplot as plt

#filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Outputs\Gulf_Stream_1\date1.npz'.replace('\\','\\')
filename = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\olci_S3A_2018-05-10 02_08_39__2018-05-10 02_05_24.npz'.replace('\\','\\')

# load npz files. encoding argument is used only if npz files have been
# saved using py2.x and are loaded by py3.x
dat = np.load(filename, encoding='latin1', allow_pickle=True)

# Retrieve dictionary
dat = dat['arr_0'].item()
matrix = np.ma.zeros((dat['Distance'].size, len(dat)-1), fill_value=-9999)
matrix_labels = [] # Name of variable

for i, key in zip(range(len(dat)), dat.keys()):
    if (type(dat[key]) is np.ma.masked_array) or (type(dat[key]) is np.ndarray): # Check if value of dictionary is np array
        matrix[:, i] = np.squeeze(dat[key])
        matrix_labels.append(key)
    
matrix = np.ma.fix_invalid(matrix)

plt.figure(figsize=(8,5))
for key in dat.keys():
    if ('SST' in key):# or ('SSHA' in key):
        plt.plot(dat['Distance'], dat[key])
        

