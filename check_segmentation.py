# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:49:52 2015

@author: ajaver
"""
import cv2
import numpy as np
import h5py
import tables
from skimage.measure import regionprops, label
from skimage import morphology
import matplotlib.pylab as plt

def analyze_image(image, bgnd_im, bgnd_mask):
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
    props_list = [coord_x, coord_y, area, perimeter, 
               major_axis, minor_axis, eccentricity, compactness, 
               orientation, solidity]
               
    return (props_list, image)
               
if __name__ == "__main__":   
#    fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';    
#    bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923.hdf5';
#    maskDB = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923_mask.db';
#    maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116-2/A002 - 20150116_140923_mask.hdf5';

    
    fileName = '/Volumes/ajaver$/DinoLite/Videos/Exp5-20150116/A002 - 20150116_140923.wmv';
    bgndFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923F.hdf5';
    featuresFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_features-2m.hdf5';
    maskFile = '/Volumes/ajaver$/DinoLite/Results/Exp5-20150116/A002 - 20150116_140923D_mask-2fn.hdf5';



#'''START TO READ THE HDF5 FROM THE PREVIOUSLY PROCESSED BGND '''    
    bgnd_fid = h5py.File(bgndFile, "r");
    bgnd = bgnd_fid["/bgnd"];
    BUFF_SIZE = bgnd.attrs['buffer_size'];
    BUFF_DELTA = bgnd.attrs['delta_btw_bgnd'];
    INITIAL_FRAME = bgnd.attrs['initial_frame'];


#'''OPEN THE VIDEO FILE AND EXTRACT SOME PARAMETERS'''        
    vid = cv2.VideoCapture(fileName)
    vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME);#initialize video to the initial position of the background buffer 
    im_width = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    im_height = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

#'''OPEN THE HDF5 TO SAVE THE PROCCESSED IMAGES (MASKS)'''
    mask_fid = h5py.File(maskFile, "r");      
    
    mask_dataset = mask_fid["/mask"];

#'''OPEN PYTABLES WITH THE STORE THE FEATURES'''    
    feature_fid = tables.open_file(featuresFile, mode = 'r')
    feature_table = feature_fid.get_node('/plate_worms')


    nChunk_prev = -1
    for ind_frame in [3000]:#range(0, 9900, 1000):#[0, 10000, 50000, 100000, 175000, 250000]:
        nChunk = np.floor(ind_frame / BUFF_DELTA)
        if nChunk_prev != nChunk:
            bgndInd = nChunk-np.ceil(BUFF_SIZE/2.0);
            if bgndInd<0:
                bgndInd = 0;
            elif bgndInd > bgnd.shape[0]:
                bgndInd = bgnd.shape-1;
            
            bgnd_im = bgnd[bgndInd,:,:];
            bgnd_mask = cv2.adaptiveThreshold(bgnd_im,1,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,61,35)
            
        vid.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, INITIAL_FRAME + ind_frame);#initialize video to the initial position of the background buffer 
        retval, image = vid.read()
        if not retval:
            break;
        
        
        data = analyze_image(image, bgnd_im, bgnd_mask)
        
        
        print bgndInd, ind_frame, len(data[0][0])
        print len(data[0][0]), sum(1 for x in feature_table.where('frame_number==%i' % ind_frame))
        
        mask = mask_dataset[ind_frame, :,:];
        
        plt.figure()
        f, (ax1, ax2) = plt.subplots(1,2)
        ax1.imshow(data[1], interpolation = 'none', cmap = 'gray')
        ax2.imshow(mask, interpolation = 'none', cmap = 'gray')
        
        
        
        
        #plt.imshow(data[1], interpolation = 'none', cmap = 'gray')
       
