# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:33:18 2015

@author: ajaver
"""
import multiprocessing as mp
import cv2
#from functools import partial

import numpy as np
from skimage.measure import regionprops, label
from skimage import morphology
import sqlite3
import Queue
import h5py

from sklearn.utils.linear_assignment_ import linear_assignment #hungarian algorithm
from scipy.spatial.distance import cdist
from collections import defaultdict
#import matplotlib.pylab as plt

def list_duplicates(seq):
    # get index of replicated elements. It only returns replicates not the first occurence.
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    
    badIndex = []
    for locs in [locs for key, locs in tally.items() 
                            if len(locs)>1]:
        badIndex += locs[1:]
    return badIndex

 
def analyze_image(parent_conn, frame, image, bgnd_im, bgnd_mask):
    image = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY);
    im_diff = np.abs(bgnd_im-image.astype(np.double));
    #mask = im_diff >20;
    #mask2 = cv2.adaptiveThreshold(image.astype(np.uint8),1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,61,35)
    mask2 = cv2.adaptiveThreshold(im_diff.astype(np.uint8),1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,61,-10)
    L = label(mask2 | bgnd_mask)
    L = morphology.remove_small_objects(L, min_size=10, connectivity=2)
    image[L==0] = 0
    
    props = regionprops(L);
    
    coord_x = [x.centroid[0] for x in props];
    coord_y = [x.centroid[1] for x in props];
    area = [x.area for x in props];
    perimeter = [x.perimeter for x in props];
    major_axis = [x.major_axis_length for x in props];
    minor_axis = [x.minor_axis_length for x in props];
    eccentricity = [x.eccentricity for x in props];
    compactness = [x.perimeter**2/x.area for x in props];
    orientation = [x.orientation for x in props];
    solidity = [x.solidity for x in props];
    
    props_list = [coord_x, coord_y, area, perimeter, 
               major_axis, minor_axis, eccentricity, compactness, 
               orientation, solidity]

    
    parent_conn.send({'frame': frame, 'image': image, 'props_list' :props_list})
    parent_conn.close()


def get_indexList(coord, coord_prev, indexListPrev, totWorms, max_allow_dist = 10.0):
    #get the indexes of the next worms from their nearest neightbors using the hungarian algorithm
    
    if coord_prev.size!=0:
        costMatrix = cdist(coord_prev, coord);
        assigment = linear_assignment(costMatrix)
        
        indexList = np.zeros(coord.shape[0]);
        speed = np.zeros(coord.shape[0])

        for row, column in assigment: #ll = 1:numel(indexList)
            if costMatrix[row,column] < max_allow_dist:
                indexList[column] = indexListPrev[row];
                speed[column] = costMatrix[row][column];
            elif column < coord.shape[0]:
                totWorms = totWorms +1;
                indexList[column] = totWorms;
        
        for rep_ind in list_duplicates(indexList):
            totWorms = totWorms +1; #assign new worm_index to joined trajectories
            indexList[rep_ind] = totWorms;
        
    else:
        indexList = totWorms + np.arange(1,coord.shape[0]+1);
        totWorms = indexList[-1]
        speed = totWorms*[None]
        #print costMatrix[-1,-1]
    return (totWorms, indexList, speed)
    
    
def analyze_frames(processes, cur, mask_dataset, coord_prev, indexListPrev, totWorms):
    #get the data from the next proccess in the queue
    dd = processes.get();
    data = dd[0].recv()
    dd[1].join()
    
    #get the indexes of the next worms from their nearest neightbors using the hungarian algorithm
    coord = np.array(data['props_list'][0:2]).T;
    totWorms, indexList, speed = get_indexList(coord, coord_prev, indexListPrev, totWorms)
    
    #put the results of the worm features in the database 
    frames = np.empty(indexList.size);
    frames.fill(data['frame']+1);
    results_list = zip(*[indexList, frames] +  data['props_list'] + [speed])
    cur.executemany('INSERT INTO plate_worms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', results_list);
    
    #save the results of the proccessed image (mask) into the hdf5
    mask_dataset.resize(ind_frame+1, axis=0); 
    mask_dataset[ind_frame,:,:] = data['image'];
    
    return [coord, indexList, totWorms]
    

N_processes = 10;
if __name__ == "__main__":   
    fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';
    bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923F.hdf5';
    maskDB = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2b.db';
    maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2b.hdf5';

    
#'''START TO READ THE HDF5 FROM THE PREVIOUSLY PROCESSED BGND '''    
    bgnd_fid = h5py.File(bgndFile, "r");
    bgnd = bgnd_fid["/bgnd"];
    BUFF_SIZE = bgnd.attrs['buffer_size'];
    BUFF_DELTA = bgnd.attrs['delta_btw_bgnd'];
    INITIAL_FRAME = bgnd.attrs['initial_frame'];

#'''START THE DATA SQL BASE TO SAVE WORM PARAMETERS'''    
    conn = sqlite3.connect(maskDB)
    cur = conn.cursor()
    cur.execute('''DROP TABLE IF EXISTS plate_worms''')
    cur.execute('''CREATE TABLE plate_worms  
                 (worm_index INT NOT NULL,
                 frame_number INT NOT NULL,
                 coord_x REAL,
                 coord_y REAL,
                 area REAL,
                 perimenter REAL,
                 major_axis REAL, 
                 minor_axis REAL,
                 eccentricity REAL, 
                 compactness REAL, 
                 orientation REAL, 
                 solidity REAL, 
                 speed REAL,
                 PRIMARY KEY (worm_index, frame_number)
                 )''') 

#'''OPEN THE VIDEO FILE AND EXTRACT SOME PARAMETERS'''        
    vid = cv2.VideoCapture(fileName)
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME);#initialize video to the initial position of the background buffer 
    im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

#'''CREATE THE HDF5 TO SAVE THE PROCCESSED IMAGES (MASKS)'''
    mask_fid = h5py.File(maskFile, "w");      
    if "mask" in mask_fid:
        del mask_fid["mask"]
    mask_dataset = mask_fid.create_dataset("/mask", (0, im_height, im_width), 
                                    dtype = "u1", 
                                chunks = (1, im_height, im_width), 
                                maxshape = (None, im_height, im_width), 
                                compression="gzip");
#'''INITIALIZE VARIABLES'''   
    processes = Queue.Queue()
    max_frame = int(BUFF_DELTA*(bgnd.shape[0] + np.ceil(BUFF_SIZE/2)));
    nChunk_prev = -1;
    coord_prev = np.empty([0]);
    totWorms = 0;
    indexListPrev = np.empty([0]);
    results_list = [];
#'''MAIN LOOP STARTS'''   
    for ind_frame in range(1000):
        nChunk = np.floor(ind_frame / BUFF_DELTA)
        if nChunk_prev != nChunk:
            bgndInd = nChunk-np.ceil(BUFF_SIZE/2);
            if bgndInd<0:
                bgndInd = 0;
            elif bgndInd > bgnd.shape[0]:
                bgndInd = bgnd.shape-1;
            
            bgnd_im = bgnd[bgndInd,:,:];
            bgnd_mask = cv2.adaptiveThreshold(bgnd_im,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,35)    
        
        nChunk_prev = nChunk;
        
        print ind_frame
        retval, image = vid.read()
        if not retval:
            break;
        
        parent_conn, child_conn = mp.Pipe();
        p = mp.Process(target=analyze_image, args=(parent_conn, ind_frame, image, bgnd_im, bgnd_mask));
        p.start();
        processes.put((child_conn, p))
        
        if processes.qsize() >= N_processes:
            coord_prev, indexListPrev, totWorms = analyze_frames(processes, cur, mask_dataset, coord_prev, indexListPrev, totWorms);
            if ind_frame % 1200 == 1:
                conn.commit();

    #proccess the last data in the queue
    for x in range(processes.qsize()):
        coord_prev, indexListPrev, totWorms = analyze_frames(processes, cur, mask_dataset, coord_prev, indexListPrev, totWorms);
    
    bgnd_fid.close()
    mask_fid.close();
    vid.release();
    conn.commit();
    conn.close();

##        processes = [mp.Process(target=test_code, args=(filename, x), name = 'P%i' % x) for x in range(N)]    
#        for p in processes:
#            p.start()
#        
#        for p in processes:
#            p.join()
#        
#        results = [output.get()[0] for p in processes]
#        print time.clock() - tic 
#    
    
#    tic = time.clock()   
#    for x in range(N):
#        aa =  test_code(filename, x);
#    results = [output.get()[0] for x in range(N)]
#    #print results
#    print time.clock() - tic
    
    
    
    #print results

# L = np.loadtxt('/Users/ajaver/python/label')
#    property_list = ['centroid', 'perimeter', 'area', 'major_axis_length', 'minor_axis_length', 'eccentricity', 'orientation', 'solidity']
#    L = np.load('/Users/ajaver/python/label.npy')
#    props = regionprops(L);
#    
#    
#    p = Pool(processes=10)
#    tic = time.clock()   
#    
#    
#    for kk in range(1): 
#    #    A = p.map(get_im_properties, property_list)
#        A = [p.apply(get_im_properties, (props, x)) for x in property_list]
#    
#    print time.clock() - tic
#        
#    
#    tic = time.clock()    
#    for kk in range(1):
#        B = get_all_properties(props)
#    print time.clock() - tic
