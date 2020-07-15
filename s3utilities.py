# -*- coding: utf-8 -*-
"""
    Created on Wed Feb  6 15:05:51 2019
    DESCRIPTION:
            The script contains some utility functions that are used to
            achieve certain tasks.
    ***IMPORTANT*** Do not use read_xml_S3_relative_orbits() function.

"""
# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import numpy as np
import os
import datetime as dt
import shutil
import xml.etree.ElementTree as ET

# =============================================================================
# BEGIN
# =============================================================================
def find_common_dates(paths):
    """
    Find products that belong to the same date, based on the sensing yyyy-mm-dd that
    is written in the name of each product's folder. Every path directory of each
    product includes S3A and S3B folders.
    
    Args:
        paths (dict) = dictionary with key being 'SRAL', 'OLCI', 'SLSTR' and
        values being string with path e.g. "D:...../SRAL"
    
    Returns:
        common_str (dict)  = dictionary with keys being 'SRAL', 'OLCI'
        and 'SLSTR', and values being lists with folder names (strings)
    """
    
    if ('SRAL' in paths.keys()) and ('SLSTR' in paths.keys()) and ('OLCI' in paths.keys()):
        path_sral = paths['SRAL']
        path_olci = paths['OLCI']
        path_slstr = paths['SLSTR']
    
        # SRAL
        folder_sral = os.listdir(path_sral)
        # Only keep S3A folders
        folder_sral = [f for f in folder_sral if (f[:3] == 'S3A') or (f[:3] == 'S3B')]
        # Keep the date (yyyymmdd) of each product (folder)
        fdate_sral = [dt.datetime.strptime(f[16:24], '%Y%m%d') for f in folder_sral]
        
        # OLCI
        folder_olci = os.listdir(path_olci)
        # Only keep S3A folders
        folder_olci = [f for f in folder_olci if (f[:3] == 'S3A') or (f[:3] == 'S3B')]
        # Keep the date (yyyymmdd) of each product (folder)
        fdate_olci = [dt.datetime.strptime(f[16:24], '%Y%m%d') for f in folder_olci]
        
        # SLSTR
        folder_slstr = os.listdir(path_slstr)
        # Only keep S3A folders
        folder_slstr = [f for f in folder_slstr if (f[:3] == 'S3A') or (f[:3] == 'S3B')]
        # Keep the date (yyyymmdd) of each product (folder)
        fdate_slstr = [dt.datetime.strptime(f[16:24], '%Y%m%d') for f in folder_slstr]
        
        # Common between SRAL and OLCI
        fdate_sral_olci = np.intersect1d(fdate_sral, fdate_olci, 
                                                             assume_unique=False,
                                                            return_indices=False)
        
        # Common between SRAL, OLCI and SLSTR
        common = np.intersect1d(fdate_sral_olci, fdate_slstr, 
                                                          assume_unique=False,
                                                            return_indices=False)
        common = [c.strftime('%Y%m%d') for c in common]
        # find common dates and save file names

        
        # Folder names for every product

        fsral_out = [item for item in folder_sral if item[16:24] in common]
        
        folci_out = [item for item in folder_olci if item[16:24] in common]
        
        fslstr_out = [item for item in folder_slstr if item[16:24] in common]
        
        common_str = {'SRAL': fsral_out,
                      'OLCI': folci_out,
                      'SLSTR': fslstr_out
                      }
        
        return common_str
    
    elif ('SRAL' in paths.keys()) and ('SLSTR' in paths.keys()):
        path_sral = paths['SRAL']
        path_slstr = paths['SLSTR']
    
        # SRAL
        folder_sral = os.listdir(path_sral)
        # Only keep S3A folders
        folder_sral = [f for f in folder_sral if (f[:3] == 'S3A') or (f[:3] == 'S3B')]
        # Keep the date (yyyymmdd) of each product (folder)
        fdate_sral = [dt.datetime.strptime(f[16:24], '%Y%m%d') for f in folder_sral]
        
        # SLSTR
        folder_slstr = os.listdir(path_slstr)
        # Only keep S3A folders
        folder_slstr = [f for f in folder_slstr if (f[:3] == 'S3A') or (f[:3] == 'S3B')]
        # Keep the date (yyyymmdd) of each product (folder)
        fdate_slstr = [dt.datetime.strptime(f[16:24], '%Y%m%d') for f in folder_slstr]
        
        # Common between SRAL and OLCI
        common = np.intersect1d(fdate_sral, fdate_slstr, 
                                                             assume_unique=False,
                                                            return_indices=False)

        common = [c.strftime('%Y%m%d') for c in common]

        fsral_out = [item for item in folder_sral if item[16:24] in common]
        
        fslstr_out = [item for item in folder_slstr if item[16:24] in common]
        
        common_str = {'SRAL': fsral_out,
                      'SLSTR': fslstr_out
                      }
        
        return common_str


def mv_folders_files(dir_src, dir_dst, file_list):
    """
    Move folders with subdirectories and files from directory to another directory
    
    Args:
        dir_src (str) = string of source directory
        dir_dst (str) = string of destination directory
        file_list (list) = list of strings with name of files
        
    Returns:
        None
    """    
    for f in os.listdir(dir_src):
        if f in file_list:
            # Create file paths
            src_file = os.path.join(dir_src, f)
            dst_file = os.path.join(dir_dst, f)
            # Move
            try:
                shutil.move(src_file, dst_file)
            except:
                print('Problematic files. Move manually:\n'+src_file)
                continue
    
    return None


def read_xml_S3(fullpath, track_direction):
    """
    Read the metadata XML file of a Sentinel-3 product and searches whether
    the orbit is ascending or descending
    
    Args:
        fullpath (str) = full directory of the XML file
        track_direction (string)= orbit with possible values being 'ascending'
        or 'descending'
    
    Returns:
        condition (bool) = True if condition is met, False if it is not met
    """
    # Parse xml file
    tree = ET.parse(fullpath)
    # Get root of the xmlTree
    root = tree.getroot()
    
    # Check if condition is true
    condition = root[1][4][0][0][0][0].attrib['groundTrackDirection'] == track_direction
    
    return condition


def read_xml_S3_relative_orbits(fullpath):
    """
    Reads XML file and records the number of the relative orbit
    
    Args:
        fullpath (str) = XML path
    
    Returns:
        out_list (list) = list with relative orbit value(s)
    """
    # Parse xml file
    tree = ET.parse(fullpath)
    
    out_list = []
    for node in tree.iter('{http://www.esa.int/safe/sentinel/1.1}relativeOrbitNumber'):
        if node.attrib['type'] == 'start':
            out_list = node.text
            
    return out_list


if __name__ == '__main__':
#    paths = {'SRAL': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL'.replace('\\', '\\'),
#         'OLCI': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\OLCI'.replace('\\', '\\'),
#         'SLSTR': r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SLSTR'.replace('\\','\\')
#         }
#    
##    common_str = find_common_dates(paths)
#    dir_src = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI'.replace('//','//')
#    dir_dst = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Gulf Stream_1\OLCI\out'.replace('//','//')
#    mv_folders_files(dir_src, dir_dst,no_data)
    path = r'D:\vlachos\Documents\KV MSc thesis\Data\Satellite\Actual_data\SRAL\S3A_SR_2_WAT____20171104T102819_20171104T111546_20171130T042207_2846_024_108______MAR_O_NT_002.SEN3'.replace('\\','\\')
    fname = 'xfdumanifest.xml'
    print(read_xml_S3(os.path.join(path,fname), 'descending'))