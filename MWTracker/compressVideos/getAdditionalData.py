"""
# -*- coding: utf-8 -*-
Created on Thu Dec 17 10:10:35 2015

@author: ajaver
"""
import os, stat
import xml.etree.ElementTree as ET
import tables
import numpy as np
import csv
from collections import OrderedDict

#%% Read/Store the XML file with the pixel size and fps info
def storeXMLInfo(info_file, masked_image_file):
    with open(info_file, 'r') as fid:
        xml_info = fid.read()
    
    root = ET.fromstring(xml_info)
    
    x_microns = float(root.findall('./info/stage/steps/equivalent/microns/x')[0].text)
    y_microns = float(root.findall('./info/stage/steps/equivalent/microns/y')[0].text)
    
    x_pixels = float(root.findall('./info/stage/steps/equivalent/pixels/x')[0].text)
    y_pixels = float(root.findall('./info/stage/steps/equivalent/pixels/y')[0].text)
    
    fps = float(root.findall('./info/camera/display/frame/rate')[0].text)
    
    pixels2microns_x = x_microns/x_pixels
    pixels2microns_y = y_microns/y_pixels    
    
    with tables.File(masked_image_file, 'r+') as fid:
        if '/xml_info' in fid: fid.remove_node('/', 'xml_info')
        xml_node = fid.create_array('/', 'xml_info', obj = bytes(xml_info, 'utf-8'))
        
        masks_node = fid.get_node('/', 'mask') 
        masks_node.attrs['fps'] = fps
        masks_node.attrs['pixels2microns_x'] = pixels2microns_x
        masks_node.attrs['pixels2microns_y'] = pixels2microns_y
    
    
def walkXML(curr_node, params = [], curr_path = ''):
    '''
    Return the structure of a ElementTree into a directory list. 
    I am not really using this function but it is cool.
    '''
    curr_path += '/' + curr_node.tag
    if len(curr_node) == 0:
        param.append((curr_path, curr_node.text))
        return params

    for node in curr_node:
        walkXML(node, params, curr_path)

#%% Read/Store the CSV file with the stage positions
def timestr2sec(timestr):
    time_parts = [float(x) for x in timestr.split(':')]
    return sum((60**ii)*part for ii,part in enumerate(time_parts[::-1]))

def getHeader(data):
    assert data    
    
    #find header (it is not always the first line)
    for ii, line in enumerate(data):
        if line[0] == 'Real Time':
            break
    assert(ii < len(line)-1)
    header = data.pop(ii)
    
    #check that the expected columns are in header
    expected_header = ['Real Time', 'Media Time', 'Location Type', \
    'Centroid/Stage/Speed X (microns[/second])', 'Centroid/Stage/Speed Y (microns[/second])', \
    'MER Min X (microns)', 'MER Min Y (microns)', 'MER Width (microns)', 'MER Height (microns)']
    assert all(col in expected_header for col in header)
    
    return header, data

def data2dict(header, data):
    assert data
    ## save data into a dictionary
    csv_dict = {}
    for ii, col_data in enumerate(zip(*data)):
        csv_dict[header[ii]] = col_data
    
    
    ## Check the data is correct
    assert all(x == 'STAGE' for x in csv_dict['Location Type'])
    del csv_dict['Location Type']
    
    for col_name in ['MER Min X (microns)', 'MER Min Y (microns)', \
    'MER Width (microns)', 'MER Height (microns)']:
        assert all(not x for x in csv_dict[col_name])
        del csv_dict[col_name]
    return csv_dict

def dict2recarray(csv_dict):
    dat = OrderedDict()
    
    dat['real_time'] = np.array([bytes(x, 'utf-8') for x in csv_dict['Real Time']])
    dat['stage_time'] = np.array([timestr2sec(x) for x in csv_dict['Media Time']])
    
    dat['stage_x'] = np.array([float(d) \
    for d in csv_dict['Centroid/Stage/Speed X (microns[/second])']])
    
    dat['stage_y'] = np.array([float(d) \
    for d in csv_dict[ 'Centroid/Stage/Speed Y (microns[/second])']])
        
    #convert into recarray (pytables friendly)
    dtype = [(kk, dat[kk].dtype) for kk in dat]
    N = len(dat['stage_x'])
    stage_recarray = np.recarray(N, dtype)
    for kk in dat: stage_recarray[kk] = dat[kk]
    
    return stage_recarray

def storeStageData(stage_file, masked_image_file):
    ## read motor data from csv
    with open(stage_file) as fid:
        reader = csv.reader(fid)
        data = [line for line in reader]
    
    #filter, check and store the data into a recarray
    header, data = getHeader(data)
    csv_dict =  data2dict(header, data)
    stage_recarray = dict2recarray(csv_dict)

    with tables.File(masked_image_file, 'r+') as mask_fid:
        if '/stage_data' in mask_fid: mask_fid.remove_node('/', 'stage_data')
        mask_fid.create_table('/', 'stage_data', obj = stage_recarray)
    
    return csv_dict

def getAdditionalFiles(video_file):
    assert(os.path.exists(video_file))
    base_name = video_file.rsplit('.')[0]
    info_file =  base_name + '.info.xml'
    stage_file = base_name + '.log.csv'
    
    #throw and exception if the additional files do not exists 
    if not (os.path.exists(info_file) and os.path.exists(stage_file)):
        raise Exception('Additional files (info.xml - log.csv) do not exists.')

    return info_file, stage_file

#%% main function to store the additional data
def storeAdditionalDataSW(video_file, masked_image_file):
    assert(os.path.exists(video_file))
    assert(os.path.exists(masked_image_file))
    
    info_file, stage_file = getAdditionalFiles(video_file)

    assert(os.path.exists(video_file))
    assert(os.path.exists(stage_file))

    #store data
    storeXMLInfo(info_file, masked_image_file)
    storeStageData(stage_file, masked_image_file)
    
    with tables.File(masked_image_file, 'r+') as mask_fid:
        mask_fid.get_node('/mask').attrs['has_finished'] = 2



    



