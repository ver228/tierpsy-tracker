# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:33:18 2015

@author: ajaver
"""
import multiprocessing as mp
import Queue
import cv2
#from functools import partial

import numpy as np
from skimage.measure import regionprops, label
from skimage import morphology
import h5py
import tables
import time

N_processes = 24;

class plate_worms(tables.IsDescription):
#class for the pytables 
    worm_index = tables.Int32Col(pos=0)
    frame_number = tables.Int32Col(pos=1)
    label_image = tables.Int32Col(pos=2)
    coord_x = tables.Float32Col(pos=3)
    coord_y = tables.Float32Col(pos=4) 
    area = tables.Float32Col(pos=5) 
    perimenter = tables.Float32Col(pos=6) 
    major_axis = tables.Float32Col(pos=7) 
    minor_axis = tables.Float32Col(pos=8) 
    eccentricity = tables.Float32Col(pos=9) 
    compactness = tables.Float32Col(pos=10) 
    orientation = tables.Float32Col(pos=11) 
    solidity = tables.Float32Col(pos=12) 
    speed = tables.Float32Col(pos=13)

def analyze_image(parent_conn, frame, image, bgnd_im, bgnd_mask):
    #segmentation procedure
    image = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY);
    im_diff = np.abs(bgnd_im-image.astype(np.double));
    #mask = im_diff >20;
    #mask2 = cv2.adaptiveThreshold(image.astype(np.uint8),1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,61,35)
    mask2 = cv2.adaptiveThreshold(im_diff.astype(np.uint8),1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,61,-10)
    L = label(mask2 | bgnd_mask)
    L = morphology.remove_small_objects(L, min_size=10, connectivity=2)
    image[L==0] = 0
    
    #extract images for each object
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
    
    N = len(props); 
    worm_index = [-1]*N #unkown
    frame_number = [frame]*N 
    label_image = range(N)
    speed = [-1]*N;
    
    props_list = [worm_index, frame_number, label_image,
                  coord_x, coord_y, area, perimeter, 
               major_axis, minor_axis, eccentricity, compactness, 
               orientation, solidity, speed] #speed
    #rearrage of the data to insert into pytables
    props_list = [tuple(x) for x in props_list]
    props_list = zip(*props_list)
     
    parent_conn.send({'frame': frame, 'image': image, 'props_list' :props_list})
    parent_conn.close()


if __name__ == "__main__":   
#    fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';    
#    bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923.hdf5';
#    maskDB = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923_mask.db';
#    maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923_mask.hdf5';

    fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';
    bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923H.hdf5';
    featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923H_features.hdf5';
    maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923H_mask.hdf5';

    #fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';
    #bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923F.hdf5';
    #featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_features-2n.hdf5';
    #maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2fn.hdf5';


#'''START TO READ THE HDF5 FROM THE PREVIOUSLY PROCESSED BGND '''    
    bgnd_fid = h5py.File(bgndFile, "r");
    bgnd = bgnd_fid["/bgnd"];
    BUFF_SIZE = bgnd.attrs['buffer_size'];
    BUFF_DELTA = bgnd.attrs['delta_btw_bgnd'];
    INITIAL_FRAME = bgnd.attrs['initial_position'];


#'''OPEN THE VIDEO FILE AND EXTRACT SOME PARAMETERS'''        
    vid = cv2.VideoCapture(fileName)
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME);#initialize video to the initial position of the background buffer 
    im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

#'''CREATE THE HDF5 TO SAVE THE PROCCESSED IMAGES (MASKS)'''
    mask_fid = h5py.File(maskFile, "w");      
    #if "mask" in mask_fid:
    #    del mask_fid["mask"]
    mask_dataset = mask_fid.create_dataset("/mask", (0, im_height, im_width), 
                                    dtype = "u1", 
                                chunks = (1, im_height, im_width), 
                                maxshape = (None, im_height, im_width), 
                                compression="gzip", shuffle=True);

#'''CREATE PYTABLES TO STORE THE FEATURES'''
    feature_fid = tables.open_file(featuresFile, mode = 'w', title = '')
    #features_group = H5features.create_group("/", "features", "Worm feature List")
    feature_table = feature_fid.create_table('/', "plate_worms", plate_worms,"Worm feature List")
    
#'''INITIALIZE VARIABLES'''   
    processes = Queue.Queue()
    max_frame = int(BUFF_DELTA*(bgnd.shape[0] + np.ceil(BUFF_SIZE/2)));
    #max_frame = 10000;    
    nChunk_prev = -1;
    coord_prev = np.empty([0]);
    totWorms = 0;
    indexListPrev = np.empty([0]);
    results_list = [];
    
    tic_first = time.time();
    tic = time.time()
#'''MAIN LOOP STARTS'''   
    for ind_frame in range(max_frame):
        nChunk = np.floor(ind_frame / BUFF_DELTA)
        if nChunk_prev != nChunk:
            bgndInd = nChunk-np.ceil(BUFF_SIZE/2.0);
            if bgndInd<0:
                bgndInd = 0;
            elif bgndInd > bgnd.shape[0]:
                bgndInd = bgnd.shape-1;
            
            bgnd_im = bgnd[bgndInd,:,:];
            bgnd_mask = cv2.adaptiveThreshold(bgnd_im,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,35)
            
            #resize the buffer everytime a new chunk is called
            N = BUFF_DELTA*(nChunk+1); 
            if N>max_frame: N = max_frame+1;
            mask_dataset.resize(N, axis=0); 
            
            feature_table.flush()
            
        nChunk_prev = nChunk;
        
        if ind_frame % 100 == 0:
            toc = time.time()
            print ind_frame, toc-tic
            tic = toc
            
        retval, image = vid.read()
        if not retval:
            break;
        
        parent_conn, child_conn = mp.Pipe();
        p = mp.Process(target=analyze_image, args=(parent_conn, ind_frame, image, bgnd_im, bgnd_mask));
        p.start();
        processes.put((child_conn, p))
        
        if processes.qsize() >= N_processes:
            dd = processes.get();
            data = dd[0].recv() #read pipe
            dd[1].join() #wait for the proccess to be completed
            
            mask_dataset[data['frame'],:,:] = data['image'];
            feature_table.append(data['props_list'])
            #coord_prev, indexListPrev, totWorms = analyze_frames(processes, mask_dataset, results_list, coord_prev, indexListPrev, totWorms);
            
    #proccess the last data in the queue
    for x in range(processes.qsize()):
        dd = processes.get();
        data = dd[0].recv()
        dd[1].join()
        
        if mask_dataset != max_frame:
            mask_dataset.resize(max_frame, axis=0); 
        
        mask_dataset[data['frame'],:,:] = data['image'];
        feature_table.append(data['props_list'])
            
        
            
        #coord_prev, indexListPrev, totWorms = analyze_frames(processes, mask_dataset, results_list, coord_prev, indexListPrev, totWorms);
    feature_table.flush()
    feature_table.cols.frame_number.create_csindex()
    print feature_table
    feature_fid.close()
    
    bgnd_fid.close()
    mask_fid.close()
    vid.release()
    
    print 'Total time %f' % (time.time() -tic_first)
    
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
