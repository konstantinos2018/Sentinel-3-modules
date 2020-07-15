# -*- coding: utf-8 -*-
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os


# =============================================================================
# BEGIN
# =============================================================================
sensor = ['SRAL', 'SLSTR', 'OLCI']
path = [r'C:\Users\vlachos\Desktop\SRAL'.replace('\\','\\'), # SRAL
        r'C:\Users\vlachos\Desktop\SLSTR'.replace('\\','\\'),# SLSTR
        r'C:\Users\vlachos\Desktop\OLCI'.replace('\\','\\')] # OLCI

fname = 'enhanced_measurement.nc' # SRAL
#fname = ['sub', 'xfdumanifest.xml'] # SLSTR, OLCI

for p, sense in zip(path, sensor):
    for subdir, dirs, files in os.walk(p):
        for f in files:
            if ('sub' in f) or ('xml' in f):
                continue
            else:
#                print(os.path.join(subdir, f))
                os.remove(os.path.join(subdir, f))
                    
#            fpath = os.path.join(os.path.join(path, f), fname)
#            os.remove(fpath)
            
#    elif (sense == 'SLSTR') or (sense == 'OLCI'):
#        # list files in folder
#        for subdir, dirs, files in os.walk(path):
#            files_list = os.listdir(os.path.join(path, f))
#            # delete file
#            for fil in files_list:
#                if 
    