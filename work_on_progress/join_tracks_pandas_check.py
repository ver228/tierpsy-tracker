
import matplotlib.pylab as plt
import numpy as np
import tables
import pandas as pd
import h5py
import cv2
import subprocess as sp
import os
##trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_CaptureTest_90pc_Ch2_18022015_230213.hdf5';
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories_mask/20150221/Trajectory_CaptureTest_90pc_Ch4_21022015_210020_short.hdf5'
#trajectories_csv = '/Volumes/behavgenom$/Avelino/movies4Andre/Trajectory_CaptureTest_90pc_Ch4_21022015_210020_short.csv'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch4_16022015_174636.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch4_16022015_174636.hdf5'

#masked_image_file = '/Volumes/behavgenom$/GeckoVideo/Compressed/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'
#trajectories_file = '/Volumes/behavgenom$/GeckoVideo/Trajectories/20150220/CaptureTest_90pc_Ch3_20022015_183607.hdf5'

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'


#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Features_Mask_short_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
feature_fid = tables.open_file(trajectories_file, mode = 'r')

df = pd.DataFrame.from_records(feature_fid.root.plate_worms[:])

#%%
joined_tracks = df[df['worm_index_joined'] > 0]


#mean_area = df_good.groupby('worm_index')['area'].aggregate(['mean', 'count'])
#valid = (mean_area['mean']>300) & (mean_area['count']>25)
#good_index = mean_area[valid].index
#
#df_good = df_good[df_good.worm_index.isin(good_index)]
#N_lim = 40
#valid = (df_good['coord_x']>N_lim) & (df_good['coord_x']<2048-N_lim) & \
# (df_good['coord_y']>N_lim) & (df_good['coord_y']<2048-N_lim) 
#df_good = df_good[valid]
##df_good = df[df['frame_number']<(25*3600) & df.worm_index.isin(good_index)]
#df_good = df_good.sort('frame_number')
tracks_data = joined_tracks[['worm_index_joined', 'frame_number', 'coord_x', \
                         'coord_y', 'area', 'major_axis', \
                         'intensity_mean']].groupby('worm_index_joined').aggregate(['mean', 'max', 'min', 'first', 'last', 'count'])

tracks_dataQ = joined_tracks[['worm_index_joined', 'frame_number', 'coord_x', \
                         'coord_y', 'area', 'major_axis', \
                         'intensity_mean']].groupby('worm_index_joined').quantile(0.9)
#%%
delX = tracks_data['coord_x']['max'] - tracks_data['coord_x']['min']
delY = tracks_data['coord_y']['max'] - tracks_data['coord_y']['min']


good_index = tracks_data[(delX>20) & (delY>20)].index
#%%
joined_tracks_good = joined_tracks[joined_tracks.worm_index_joined.isin(good_index)]
track_data = joined_tracks_good[['worm_index_joined', 'frame_number']].groupby('worm_index_joined').aggregate(['max', 'min','count'])
track_size = (track_data['frame_number']['max']-track_data['frame_number']['min']+1)
track_size.sort(ascending=False)
#track_joined = joined_tracks_good['worm_index_joined'].value_counts()
#joined_tracks_good.to_csv(trajectories_csv, index=False)

#%%
worm_ind = joined_tracks_good[joined_tracks_good['worm_index_joined']==track_size.index[4]]

plt.figure()
plt.plot(worm_ind['coord_x'],worm_ind['coord_y'])
plt.axis('equal');

mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]
imsize = mask_dataset.shape[1:];


max_xsize = np.max(worm_ind['bounding_box_xmax'] -  worm_ind['bounding_box_xmin']) + 20
max_ysize = np.max(worm_ind['bounding_box_ymax'] -  worm_ind['bounding_box_ymin']) + 20

all_worm = np.zeros((len(worm_ind)+200, max_ysize, max_xsize), dtype = np.uint8);
for ii in range(len(worm_ind)+200):
    
    ii_l = min(max(100, ii), len(worm_ind)-1);
    w = worm_ind.iloc[ii_l-100]
    
    range_x = [max(int(np.floor(w['coord_x'] - max_xsize/2)), 0), min(int(np.ceil(w['coord_x'] + max_xsize/2)), imsize[1])]
    delX = range_x[1]-range_x[0]
    if delX != max_xsize: range_x[-1] += max_xsize-delX;
    
    range_y = [max(int(np.floor(w['coord_y'] - max_ysize/2)), 0), min(int(np.ceil(w['coord_y'] + max_ysize/2)), imsize[0])]
    delY = range_y[1]-range_y[0]
    if delY != max_xsize: range_y[-1] += max_ysize-delY;
    
    
    t = w['frame_number']-1
    if ii <100:
        t -= 100-ii;
    elif ii>len(worm_ind)-1:
        t += ii-len(worm_ind)+1;

    img_worm = mask_dataset[t, range_y[0]:range_y[1], range_x[0]:range_x[1] ]
    
    #bin_worm = (img_worm<w['threshold']/0.95)&(img_worm!=0)
    bin_worm = (img_worm<w['threshold'])&(img_worm!=0)
    
    bin_worm = cv2.morphologyEx(bin_worm.astype(np.uint8), cv2.MORPH_CLOSE,np.ones((3,3)))
    #bin_worm = cv2.erode(bin_worm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=2)
    
    #np.mean(img_worm[(img_worm!=0) & (img_worm>thresh)] )
    img_worm[img_worm==0] = 210;
    
    bin_border = cv2.morphologyEx(bin_worm.astype(np.uint8), cv2.MORPH_GRADIENT,np.ones((2,2)))
    img_worm[bin_border!=0] = 255;
    img_worm[0] = 0;
    if img_worm.shape == all_worm.shape[1:3]:
        all_worm[ii,:,:] = img_worm
    print ii
    #plt.figure()
    #plt.imshow(img_worm, interpolation = 'none', cmap = 'gray')

import tifffile
tifffile.imsave('test.tiff', all_worm, compress=4) 
#%%
#save_video_file = 'test.avi';
#    
##parameters to minimize the video using ffmpeg
#command = [ 'ffmpeg',
#        '-y', # (optional) overwrite output file if it exists
#        '-f', 'rawvideo',
#        '-vcodec','rawvideo',
#        '-s', '%ix%i' % (max_ysize, max_xsize), # size of one frame
#        '-pix_fmt', 'gray',
#        '-r', '25', # frames per second
#        '-i', '-', # The imput comes from a pipe
#        '-an', # Tells FFMPEG not to expect any audio
#        '-vcodec', 'mjpeg',
#        '-threads', '0',
#        '-qscale:v', '0',
#        save_video_file]
#devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
#pipe.stdin.write(img_worm.tostring())
#pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
#pipe.terminate()
#%%
#    #total number of frames.
#    tot_frames = float(I_worms.shape[0])
#    initial_time = fps_time = time.time()
#    last_frame = 0;
#    
#    
#    for frame_number in range(0,I_worms.shape[0],n_frames_jumped):
#        pipe.stdin.write(I_worms[frame_number,:,:].tostring()) #write frame 
#        
#        #calculate progress
#        if frame_number%1000 == 0:
#            time_str = str(datetime.timedelta(seconds=round(time.time()-initial_time)))
#            fps = (frame_number-last_frame+1)/(time.time()-fps_time)
#            progress_str = 'Downsampling video. Total time = %s, fps = %2.1f; %3.2f%% '\
#                % (time_str, fps, frame_number/tot_frames*100)
#            
#            sendQueueOrPrint(status_queue, progress_str, base_name);
#            fps_time = time.time()
#            last_frame = frame_number;
#
#    #close files
#    pipe.terminate()
#    mask_fid.close()

#image_buffer = mask_dataset[frame_number:(frame_number+buffer_size),:,:]
#%%


#fig = []
#fig.append(plt.figure())
#fig.append(plt.figure())
#fig.append(plt.figure())
#for ind in track_joined.index:
#    delX = tracks_data['coord_x']['max'][ind] - tracks_data['coord_x']['min'][ind]
#    delY = tracks_data['coord_y']['max'][ind] - tracks_data['coord_y']['min'][ind]
#    
#    if (delX>20) or (delY > 20):
#        coord = joined_tracks[joined_tracks['worm_index_joined'] == ind]
#        xx = np.array(coord['coord_x'])
#        yy = np.array(coord['coord_y'])
#        tt = np.array(coord['frame_number'])
#        plt.figure(fig[0].number)
#        plt.plot(tt,xx)
#        plt.figure(fig[1].number)
#        plt.plot(tt,yy)
#        plt.figure(fig[2].number)
#        plt.plot(xx,yy)
#
#plt.figure(fig[2].number)        
#plt.plot((0,0,2048,2048,0), (0,2048,2048,0,0), 'k')
#plt.xlim((0,2048))
#plt.ylim((0,2048))


#    if tracks_data['coord_x']>2.2:
#        coord = joined_tracks[joined_tracks['worm_index_joined'] == ind]
#        xx = np.array(coord['coord_x'])
#        yy = np.array(coord['coord_y'])
#        tt = np.array(coord['frame_number'])
        #len(xx)
        #plt.plot(xx,yy)
        


#    w_ind_list = coord['worm_index'].unique()
#    for w_ind in w_ind_list:
#        plt.plot(tracks_data['coord_x']['last'][w_ind], \
#        tracks_data['coord_y']['last'][w_ind], 'r.')
#        plt.plot(tracks_data['coord_x']['first'][w_ind], \
#        tracks_data['coord_y']['first'][w_ind], 'g.')