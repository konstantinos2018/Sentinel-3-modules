# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:45:19 2018

@author: vlachos
"""
# =============================================================================
#  IMPORTS
# =============================================================================
# Python Modules
import os
import sys
import zipfile as zp
    

def unzpdel(zippathin, zippathout, nclist=[], rmv=False, sensor=None):
    """
    Extracts files from a folder which contains zip files only.
    Input:
        zippath = path where the zip files are located
        nclist = the list of strings of the specific files that are needed to be extracted
        rmv = boolean (True/False), whether to delete the zip files after extraction or not
        sensor = string, a name of the sensor (i.e. OLCI, SLSTR or SRAL)
    Output:
        None
    """    
    # check if sensor name is valid
    print('Checking Inputs...')
    if (sensor!='SRAL') and (sensor!='OLCI') and (sensor!='SLSTR'):
        raise ValueError('sensor name not acceptable')
    print('Checked')
    
    # List files in given path
    flist = os.listdir(zippathin)
    
    counter = 0
    n = len(flist)
    lst = [] # initialize list of bad zip files
    
    print('Start processing...')
    if (sensor=='SRAL') or (sensor=='OLCI'):
    # For every file in directory
        for fname in flist:
            
            # progress
            sys.stdout.write("\rProgress...{0}%".format((float(counter)/n)*100))
            sys.stdout.flush()
            
            # Open zip file
            try:
                zfile = zp.ZipFile(zippathin + fname, 'r')
                
                # list directories inside zip file
                znamelist = zfile.namelist()
                
                for item in nclist:
                    for zname in znamelist:
                    # Check if item is in the zname
                        if item in zname:
                            zfile.extract(member=zname, path=zippathout)
                
                # Close zip file
                zfile.close()
                    
                # Check whether to delete zip files or not
                if rmv==True:
                    # Delete zip file from folder
                    os.remove(zippathin + fname)
            except:
                lst.append(fname)
                continue
            
            counter = counter + 1
            
    elif sensor=='SLSTR':
        # For every file in directory
        for fname in flist:
            
            # progress
            sys.stdout.write("\rProgress...{0}%".format((float(counter)/n)*100))
            sys.stdout.flush()
            
            # Open zip file
            try:
                zfile = zp.ZipFile(zippathin + fname, 'r')

                # list directories inside zip file
                znamelist = zfile.namelist()
                
     
                for zname in znamelist:
                    zfile.extract(member=zname, path=zippathout)
    
                # Close zip file
                zfile.close()
                    
                # Check whether to delete zip files or not
                if rmv==True:
                    # Delete zip file from folder
                    os.remove(zippathin + fname)
            except:
                lst.append(fname)
                continue
            
            counter = counter + 1
    
    print('Processing ended')
    print('Writing errors log file...')
    # Write error_log text file
    f_obj = open(os.path.join(zippathout, 'error_log_'+sensor+'.txt'), 'w')
    for i in lst:
        f_obj.write('{0}\n'.format(i[:-4]))
    f_obj.close()
    
    print('Errors log files written')
    return None

if __name__=='__main__':

    products = ['OLCI', 'SLSTR', 'SRAL']
    for sensor in products:
		# SRAL
		if sensor == 'SRAL':
			zippathin = 'C:\\Users\\vlachos\\Desktop\\SRAL\\'
			zippathout = 'C:\\Users\\vlachos\\Desktop\\SRAL\\'
			
			nclist = ['enhanced_measurement.nc', 'xfdumanifest.xml']
		# SLSTR
		elif sensor == 'SLSTR':
			zippathin = 'C:\\Users\\vlachos\\Desktop\\SLSTR\\'
			zippathout = 'C:\\Users\\vlachos\\Desktop\\SLSTR\\'
			
			nclist = []
				
		# OLCI
		elif sensor == 'OLCI':
			zippathin = 'C:\\Users\\vlachos\\Desktop\\OLCI\\'
			zippathout = 'C:\\Users\\vlachos\\Desktop\\OLCI\\'
			
			nclist = ['chl_nn.nc', 'chl_oc4me.nc', 'geo_coordinates.nc', 'instrument_data.nc',
					  'iop_nn.nc', 'iwv.nc', 'par.nc', 'tie_geometries.nc',
					  'tie_geo_coordinates.nc', 'tie_meteo.nc', 'time_coordinates.nc',
					  'trsp.nc', 'tsm_nn.nc', 'wqsf.nc', 'w_aer.nc', 'xfdumanifest.xml']

    # Files we want to keep
#    nclist = []


#    
		unzpdel(zippathin=zippathin, zippathout=zippathout, nclist=nclist, rmv=True, sensor=sensor)