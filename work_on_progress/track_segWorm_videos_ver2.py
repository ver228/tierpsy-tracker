# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 10:16:47 2015

@author: ajaver
"""
import os
import subprocess as sp
import tables
import h5py
import pandas as pd
import cv2
#import statsmodels.api as sm
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import numpy as np
import time

#import matplotlib
#matplotlib.use("WAgg") #generate a png as output by default
import matplotlib.pylab as plt
#import matplotlib.animation as animation

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/Trajectory_CaptureTest_90pc_Ch2_18022015_230213.hdf5'
#save_dir = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/'

#masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431.hdf5'
#segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/CaptureTest_90pc_Ch1_02022015_141431_segworm.hdf5'
#save_dir = '/Users/ajaver/Desktop/Gecko_compressed/prueba/trajectories/worms/'

masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/Capture_Ch4_23032015_111907.hdf5'
trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch4_23032015_111907.hdf5'
segworm_file = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/Capture_Ch4_23032015_111907_segworm.hdf5'
save_dir = '/Users/ajaver/Desktop/Gecko_compressed/20150323/trajectories/worms_Ch4_kezhi/'


# These are the "Tableau 20" colors as RGB.  
#tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
#             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
#             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
#             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
#             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
#for i in range(len(tableau20)):  
#    r, g, b = tableau20[i]  
#    tableau20[i] = (r / 255., g / 255., b / 255.) 

#from colorbrewer
Dark2 = {
    3: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)'],
    4: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)'],
    5: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)', 'rgb(102,166,30)'],
    6: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)', 'rgb(102,166,30)', 'rgb(230,171,2)'],
    7: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)', 'rgb(102,166,30)', 'rgb(230,171,2)', 'rgb(166,118,29)'],
    8: ['rgb(27,158,119)', 'rgb(217,95,2)', 'rgb(117,112,179)', 'rgb(231,41,138)', 'rgb(102,166,30)', 'rgb(230,171,2)', 'rgb(166,118,29)', 'rgb(102,102,102)']
}

colorpalette = [Dark2[5][x] for x in [0,1,3]];
colorpalette = [x.split(')')[0].split('(')[1].split(',') for x in colorpalette]
for i,x in  enumerate(colorpalette):
    colorpalette[i] = tuple([int(y) for y in x])


class writeVideoffmpeg:
    def __init__(self, file_name, width = 100, height = 100):
        command = [ 'ffmpeg',
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '%ix%i' % (width,height), # size of one frame
        '-pix_fmt', 'rgb24',
        '-r', '25', # frames per second
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'mjpeg',
        '-threads', '0',
        '-qscale:v', '0',
        file_name]
        
        devnull = open(os.devnull, 'w')  #use devnull to avoid printing the ffmpeg command output in the screen
        self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=devnull)
        
    def write(self, image):
        self.pipe.stdin.write(image.tostring())
        #self.pipe.stdin.write(image)
    
    def release(self):
        self.pipe.terminate()




if not os.path.exists(save_dir):
    os.mkdir(save_dir)


table_fid = pd.HDFStore(trajectories_file, 'r');
df = table_fid['/plate_worms'];
df =  df[df['worm_index_joined'] > 0]
good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
df = df[df.worm_index_joined.isin(good_index)];
table_fid.close()


#smoothed_CM = {'coord_x':np.empty(0), 'coord_y':np.empty(0), \
#                'frame_number':np.empty(0), 'worm_index':np.empty(0)}
WINDOW_SIZE = 101;
smoothed_CM = {};
for worm_index in good_index:
    dat = df[df['worm_index_joined']==worm_index][['coord_x', 'coord_y', 'frame_number']]
    x = np.array(dat['coord_x']);
    y = np.array(dat['coord_y']);
    t = np.array(dat['frame_number']);
    
    
    first_frame = np.min(t);
    last_frame = np.max(t);
    tnew = np.arange(first_frame, last_frame+1);
    if len(tnew) <= WINDOW_SIZE:
        continue
    
    fx = interp1d(t, x)
    xnew = savgol_filter(fx(tnew), 101, 3);
    fy = interp1d(t, y)
    ynew = savgol_filter(fy(tnew), 101, 3);
    
    smoothed_CM[worm_index] = {}
    smoothed_CM[worm_index]['coord_x'] = xnew
    smoothed_CM[worm_index]['coord_y'] = ynew
    smoothed_CM[worm_index]['first_frame'] = first_frame
    smoothed_CM[worm_index]['last_frame'] = last_frame
    

#open the hdf5 with the masked images
mask_fid = h5py.File(masked_image_file, 'r');
mask_dataset = mask_fid["/mask"]


segworm_fid = h5py.File(segworm_file, 'r')
#segworm_table  = results_fid.get_node('/segworm_results')


figure_list = {};

ROI_SIZE = 130;
MOVIE_SCALE = 1#1.5;
tic = time.time()
tic_first = tic
for frame in range(0, df['frame_number'].max()):
    
    img = mask_dataset[frame,:,:]
    
    index2search = []
    for worm_index in smoothed_CM:
        if (frame >= smoothed_CM[worm_index]['first_frame']) and \
        (frame <= smoothed_CM[worm_index]['last_frame']):
            index2search.append(worm_index)
    
    wormsInFrame = df[df['frame_number']==frame]
    for worm_index in index2search:
        
        if frame == smoothed_CM[worm_index]['first_frame']:
            #intialize figure and movie recorder
            figure_list[worm_index] = {};
            #figure_list[worm_index]['fig'] = plt.figure(figsize=(2.5, 2.5), dpi=80, facecolor=None)
            #w,h = figure_list[worm_index]['fig'].canvas.get_width_height()
            movie_save_name = save_dir + ('worm_%i.avi' % worm_index)  
            figure_list[worm_index]['writer'] = writeVideoffmpeg(movie_save_name, width=ROI_SIZE*MOVIE_SCALE, height=ROI_SIZE*MOVIE_SCALE)
    
        ind = int(frame-smoothed_CM[worm_index]['first_frame'])
        range_x = np.round(smoothed_CM[worm_index]['coord_x'][ind]) + [-ROI_SIZE/2, ROI_SIZE/2]
        range_y = np.round(smoothed_CM[worm_index]['coord_y'][ind]) + [-ROI_SIZE/2, ROI_SIZE/2]
        
        if range_x[0]<0: range_x -= range_x[0]
        if range_y[0]<0: range_y -= range_y[0]
        
        if range_x[1]>img.shape[1]: range_x += img.shape[1]-range_x[1]-1
        if range_y[1]>img.shape[0]: range_y += img.shape[0]-range_y[1]-1
        
        worm_img =  img[range_y[0]:range_y[1], range_x[0]:range_x[1]]
        
        
        worm = wormsInFrame[wormsInFrame['worm_index_joined'] == worm_index]
        threshold = worm['threshold'].values
        segworm_id = worm['segworm_id'].values
        worm_rgb= cv2.cvtColor(worm_img, cv2.COLOR_GRAY2RGB);
        
        is_draw_contour = False
        if is_draw_contour:
            if (len(threshold) == 1) and (segworm_id < 0):
                worm_mask = ((worm_img<threshold)&(worm_img!=0)).astype(np.uint8)        
                worm_mask = cv2.morphologyEx(worm_mask, cv2.MORPH_CLOSE,np.ones((3,3)))
                worm_rgb[:,:,1][worm_mask!=0] = 150
            
            worm_rgb = cv2.resize(worm_rgb,(0,0),fx=MOVIE_SCALE, fy=MOVIE_SCALE);
            
            if (len(segworm_id)==1) and (segworm_id >= 0):
                strC = [2, 5, 10]
                for ii, key in enumerate(['contour_ventral', 'skeleton', 'contour_dorsal']):
                    dat = np.squeeze(segworm_fid['/segworm_results/' + key][segworm_id,:,:]);
                    #dat = np.squeeze(segworm_table[segworm_id][key])
                    xx = (dat[1,:]-range_x[0])*MOVIE_SCALE;
                    yy = (dat[0,:]-range_y[0])*MOVIE_SCALE;
                    pts = np.round(np.vstack((xx,yy))).astype(np.int32).T
                    cv2.polylines(worm_rgb, [pts], False, colorpalette[ii], thickness=1, lineType = 8)
                    #plt.plot(xx,yy, color=colorpalette[ii])
                cv2.circle(worm_rgb, tuple(pts[0]), 2, (225,225,225), thickness=-1, lineType = 8)
                cv2.circle(worm_rgb, tuple(pts[0]), 3, (0,0,0), thickness=1, lineType = 8)
        else:
            worm_rgb = cv2.resize(worm_rgb,(0,0),fx=MOVIE_SCALE, fy=MOVIE_SCALE);
        #WRITE FRAME TO THE VIDEO
        figure_list[worm_index]['writer'].write(worm_rgb)
        
    if frame == smoothed_CM[worm_index]['last_frame']:
        figure_list[worm_index]['writer'].release()    
 
        
    if frame%25 == 0:
        print frame, time.time() - tic
        tic = time.time()

mask_fid.close()
segworm_fid.close()  

#plt.close('all')  
for worm_index in figure_list:
    figure_list[worm_index]['writer'].release()    

