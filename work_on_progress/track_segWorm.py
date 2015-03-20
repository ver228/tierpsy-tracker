# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
#import matplotlib.pylab as plt
import numpy as np
import h5py
import cv2
import time
import tables
import matplotlib.pylab as plt

if not 'eng' in globals():
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('/Users/ajaver/GitHub_repositories/SegWorm/Only_segWorm'));
    eng.warning('off', 'normWorms:VulvaContourTooShort')
    eng.warning('off', 'normWorms:NonVulvaContourTooShort')


RESAMPLING_NUM = 65.0
class segworm_results(tables.IsDescription):
#class for the pytables 
    plate_worms_id = tables.Int32Col(pos=0)
    worm_index_joined = tables.Int32Col(pos=1)
    frame_number = tables.Int32Col(pos=2)
    skeleton = tables.Float32Col(shape = (RESAMPLING_NUM,2), pos=3)
    contour_ventral = tables.Float32Col(shape = (RESAMPLING_NUM,2), pos=4)
    contour_dorsal = tables.Float32Col(shape = (RESAMPLING_NUM,2), pos=5)
    
#import StringIO
#out = StringIO.StringIO()

#masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch3_21022015_210020.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch3_21022015_210020.hdf5'

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/aCaptureTest_90pc_Ch1_02022015_141431.hdf5'


#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]


#read that frame an select trajectories that were considered valid by join_trajectories
table_fid = pd.HDFStore(trajectories_file, 'r')
df = table_fid['/plate_worms']
df =  df[df['worm_index_joined'] > 0]

tracks_data = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']].groupby('worm_index_joined').aggregate(['max', 'min', 'count'])

#filter for trajectories that move too little (static objects)
MIN_DISPLACEMENT = 20;
ROI_SIZE = 130;

delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']

good_index = tracks_data[(delX>MIN_DISPLACEMENT) & (delY>MIN_DISPLACEMENT)].index
df = df[df.worm_index_joined.isin(good_index)]
table_fid.close()

#df['segworm_results_id'] = pd.Series(-1, index = df.index)


#calculate track length, it is important to do this instead of counting because some joined tracks are discontinous
#for the moment usesless
track_size = (tracks_data.loc[good_index]['frame_number']['max']- \
    tracks_data.loc[good_index]['frame_number']['min']+1)


#open the file again, this time using pytables in append mode to add segworm data
results_fid = tables.open_file(trajectories_file, 'r+')
if 'segworm_results' in results_fid.root._v_children.keys():
    results_fid.remove_node('/segworm_results')
segworm_table = results_fid.create_table('/', "segworm_results", segworm_results,"Results from the skeletonization using segWorm.")
#segworm_results = results_fid.create_vlarray(results_fid.root, 'segworm_results',
#tables.ObjectAtom(), "", filters=tables.Filters(complevel = 1, complib = 'blosc', shuffle = True))

tracking_table = results_fid.get_node('/plate_worms')


prev_worms = {}

tic = time.time()
tic_first = tic
for frame in range(0, 100):#df['frame_number'].max()):
    
    img = mask_dataset[frame,:,:]
    
    for (ii, worm) in df[df.frame_number==frame+1].iterrows():
        
        worm_index = int(worm['worm_index_joined']);    
        #initialize dictionary
        if not worm_index in prev_worms.keys():
            prev_worms[worm_index] = [];
            
        range_x = np.round(worm['coord_x']) + [-ROI_SIZE/2, ROI_SIZE/2]
        range_y = np.round(worm['coord_y']) + [-ROI_SIZE/2, ROI_SIZE/2]
        
        if (range_y[0] <0) or (range_y[0]>= img.shape[0]) or \
        (range_x[0] <0) or (range_x[0]>= img.shape[1]):
            continue
        
        worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
        
        worm_mask = ((worm_img<worm['threshold'])&(worm_img!=0)).astype(np.uint8)        
        worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
        
        if prev_worms[worm_index] and (frame - prev_worms[worm_index]['frame'] > 5):
            
            prev_worms[worm_index] = []
        
        #it is much faster to pass the data as a bytearray rather than create a matlab object using matlab.uit8
        
        worm_results, worm_struct = eng.getWormSkeleton(bytearray(worm_img), bytearray(worm_mask), \
        worm_mask.shape[0], worm_mask.shape[1], frame, prev_worms[worm_index], RESAMPLING_NUM, nargout=2);
        prev_worms[worm_index] = worm_results
        
        if not eng.isempty(worm_results):
            offset_data = {}
            for key in {'dorsal', 'ventral', 'skeleton'}:
                offset_data[key] = np.abs(np.array(worm_results[key])).astype(np.int)-1;
                offset_data[key][:,0] += range_x[0];
                offset_data[key][:,1] += range_y[0];
            
            segworm_table.append([(ii, worm_index, frame, offset_data['skeleton'], \
                offset_data['dorsal'], offset_data['ventral'])])
            tracking_table.cols.segworm_id[ii] = segworm_table.nrows-1;
    
    results_fid.flush()
    if frame%25 == 0:
        print frame, time.time() - tic
        tic = time.time()
    
mask_fid.close()
results_fid.close()
eng.quit()


#if not worm_index in video_fid.keys():
#            prev_worms[worm_index] = [];
#            video_fid[worm_index] = \
#                writeVideoffmpeg('%sworm_%i.avi'% (save_dir, worm_index), ROI_SIZE,ROI_SIZE)

#add data frame into the result struct


#        img_color = cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
#        img_color[:,:,2][worm_mask!=0] = 255
#        
#        
#        plt.figure()
#        plt.imshow(img_color, interpolation='none', cmap = 'gray')
#        if not eng.isempty(worm_results):
#            skeleton = np.abs(np.array(worm_results['skeleton'])).astype(np.int)-1
#            ventral = np.abs(np.array(worm_results['ventral'])).astype(np.int)-1
#            dorsal = np.abs(np.array(worm_results['dorsal'])).astype(np.int)-1
#            
#            plt.plot(skeleton[:,0],skeleton[:,1], 'b')
#            plt.plot(ventral[:,0],ventral[:,1], 'g')
#            plt.plot(dorsal[:,0],dorsal[:,1], 'r')
#            plt.plot(dorsal[0,0],dorsal[0,1], 'or')
#            plt.plot(dorsal[-1,0],dorsal[-1,1], 'sb')
#            print worm_struct['orientation']
#            
#        plt.xlim([0,ROI_SIZE])
#        plt.ylim([0,ROI_SIZE])
#        plt.axis('off')
#            
            
#
#for fid in video_fid:
#    video_fid[fid].release()
#print time.time() - tic_first

##############

#plt.imshow(cv2.Canny(worm_imgS,90,30), cmap='gray', interpolation='none')


#        pix_count = np.bincount(worm_img.flat)
#        pix_count[0] = 0;
#        pix_weight = pix_count*np.arange(pix_count.size)
#        
#        from skimage.filter import threshold_otsu
#        otsu_thresh = threshold_otsu(worm_img[worm_img!=0])
#        cumhist = np.cumsum(pix_count);
#        
#        cumhist[(otsu_thresh-2):(otsu_thresh+3)]
#        
#        
#        xx_t = np.arange((otsu_thresh-2),(otsu_thresh+3));
#        yy_t = cumhist[xx_t]
#        pp = np.polyfit(xx_t, yy_t, 1)
#
#        #xx = np.arange(otsu_thresh-50, cumhist.size)
#        xx = np.arange(otsu_thresh, cumhist.size)
#        yy = np.polyval(pp, xx)
#        
#        plt.figure()
#        plt.plot(cumhist)
#        plt.plot(otsu_thresh, cumhist[otsu_thresh], 'xr')
#        plt.plot(xx,yy, 'g')
#       
#        thresh = np.where((cumhist[xx]-yy)/yy>0.02)[0][0] + otsu_thresh
#        
        #plt.figure()
        #plt.imshow(worm_img<thresh, interpolation='none', cmap = 'gray')
        #worm_img = cv2.bilateralFilter(worm_img,-1, 5, 5)
        

            
#            worm_mask = ((worm_img<thresh)&(worm_img!=0)).astype(np.uint8)
#            worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
#            eng.workspace['mask']  = matlab.uint8(list(worm_mask.flat), worm_mask.shape[::-1])
#            worm_resuts = eng.eval('segWormBWimg(mask, 1, 0.1, 0);');
#            skeleton = np.array(worm_resuts['skeleton']['pixels'])
#            contour = np.array(worm_resuts['contour']['pixels'])
            
#            plt.figure()
#            plt.imshow(worm_img, interpolation='none', cmap = 'gray')
#            plt.plot(skeleton[:,0]-1,skeleton[:,1]-1, 'r')
#            plt.plot(contour[:,0]-1,contour[:,1]-1, 'g')
            
            
        #all_skeletons[worm_index].append(skeleton)
        
        
        

        #worm_resuts = eng.workspace['worm']
        #eng.eval('imshow(mask, [])')
        
        #plt.figure()
        #plt.imshow(worm_mask, interpolation='none', cmap = 'gray')




#        
#    
#    bin_worm = cv2.morphologyEx(bin_worm.astype(np.uint8), cv2.MORPH_CLOSE,np.ones((3,3)))
    #bin_worm = cv2.erode(bin_worm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=2)
    
#    #np.mean(img_worm[(img_worm!=0) & (img_worm>thresh)] )
#    img_worm[img_worm==0] = 210;
#    
#    bin_border = cv2.morphologyEx(bin_worm.astype(np.uint8), cv2.MORPH_GRADIENT,np.ones((2,2)))
#    img_worm[bin_border!=0] = 255;
#    img_worm[0] = 0;
#    
#    plt.figure()
#    plt.imshow(img_worm, cmap='gray', interpolation='none')
#    


#df = pd.DataFrame.from_records(feature_fid.root.plate_worms[:])
#
##%%
#joined_tracks = df[df['worm_index_joined'] > 0]
