# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:39:41 2015

@author: ajaver
"""
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
import h5py
import cv2
import time
import os
import subprocess as sp

#import matlab.engine
#eng = matlab.engine.start_matlab()
#eng.addpath('/Users/ajaver/GitHub_repositories/SegWorm/Only_segWorm');

#masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#save_dir = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/'

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch3_21022015_210020.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch3_21022015_210020.hdf5'
save_dir = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch3_21022015_210020_'


#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]


#read that frame an select trajectories that were considered valid by join_trajectories
df = pd.HDFStore(trajectories_file, 'r')['/plate_worms']
#df =  df[df['worm_index_joined'] > 0]


tracks_data = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']].groupby('worm_index_joined').aggregate(['max', 'min', 'count'])

#filter for trajectories that move too little (static objects)
MIN_DISPLACEMENT = 20;
delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']

good_index = tracks_data[(delX>MIN_DISPLACEMENT) & (delY>MIN_DISPLACEMENT)].index
df = df[df.worm_index_joined.isin(good_index)]

#calculate track length, it is important to do this instead of counting because some joined tracks are discontinous
#for the moment usesless
track_size = (tracks_data.loc[good_index]['frame_number']['max']- \
    tracks_data.loc[good_index]['frame_number']['min']+1)

video_fid = {}
tic_first = time.time()
for frame in range(0, 10):#df['frame_number'].max()):
    print frame
    img = mask_dataset[frame,:,:]
    
    for (ii, worm) in df[df.frame_number==frame+1].iterrows():
        
        worm_index = int(worm['worm_index_joined']);    
        #initialize dictionary
        if not worm_index in video_fid.keys():
            #all_images[worm_index] = [];
            
            command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '%ix%i' % (100,100), # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '5', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mjpeg',
            '-threads', '0',
            '-qscale:v', '0',
            '%sworm_%i.avi'% (save_dir, worm_index)]
    
    
            devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
            video_fid[worm_index] = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
        
#        range_x = [int(worm['bounding_box_xmin']-30), int(worm['bounding_box_xmax']+30)]
#        range_y = [int(worm['bounding_box_ymin']-30), int(worm['bounding_box_ymax']+30)]
        
        
        #range_x = [int(worm['bounding_box_xmin']-30), int(worm['bounding_box_xmax']+30)]
        #range_y = [int(worm['bounding_box_ymin']-30), int(worm['bounding_box_ymax']+30)]
        
        range_x = np.round(worm['coord_x']) + [-50, 50]
        range_y = np.round(worm['coord_y']) + [-50, 50]
        
        if (range_y[0] <0) or (range_y[0]>= img.shape[0]) or \
        (range_x[0] <0) or (range_x[0]>= img.shape[1]):
            continue
        
        worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
        
        worm_mask = ((worm_img<worm['threshold'])&(worm_img!=0)).astype(np.uint8)        
        worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
        
        #tic = time.time();
        #eng.workspace['mask']  = matlab.uint8(list(worm_mask.flat), worm_mask.shape[::-1])
        #print time.time()-tic;

        tic = time.time();
        #it is much faster to pass the data as a bytearray rather than create a matlab object using matlab.uit8
        (worm_resuts, err_num, err_msg) = eng.segWormBWimg(bytearray(worm_mask), worm_mask.shape[0], worm_mask.shape[1], 1, 0.1, 0, nargout=3)#, stdout=out)
        print time.time()-tic;
        if not eng.isempty(worm_resuts):
            skeleton = np.array(worm_resuts['skeleton']['pixels']).astype(np.int)-1
            contour = np.array(worm_resuts['contour']['pixels']).astype(np.int)-1
            
            worm_rgb = cv2.cvtColor(worm_img,cv2.COLOR_GRAY2RGB)
            worm_rgb[contour[:,1],contour[:,0],:] = [0,128,0]
            worm_rgb[skeleton[:,1],skeleton[:,0],:] = [200,0,0]
            video_fid[worm_index].stdin.write(worm_rgb.tostring())
#            all_images[worm_index].append(worm_rgb)
#            
            plt.figure()
            plt.imshow(worm_img, interpolation='none', cmap = 'gray')
            plt.plot(skeleton[:,0],skeleton[:,1], 'r')
            plt.plot(contour[:,0],contour[:,1], 'g')
##        
for fid in video_fid:
    video_fid[fid].terminate()
print time.time() - tic_first
#eng.quit()

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
