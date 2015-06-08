# -*- coding: utf-8 -*-
"""
Created on Sat May 16 12:18:53 2015

@author: ajaver
"""
import tables
import pandas as pd
import numpy as np
import cv2
import os
import sys
import subprocess as sp
sys.path.append('../videoCompression/')
from parallelProcHelper import timeCounterStr

class writeVideoffmpeg:
    def __init__(self, file_name, width = 100, height = 100, pix_fmt = 'gray'):
        #use pix_fmt = rgb24 for color images
        command = [ 'ffmpeg',
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '%ix%i' % (width,height), # size of one frame
        '-pix_fmt', pix_fmt,
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



def drawTrajectoriesVideo(masked_image_file, trajectories_file, 
                          max_track_draw = 100, n_frames_jumped = 25, movie_scale = 0.25):
    
    
    base_name = masked_image_file.rpartition('.')[0].rpartition(os.sep)[-1]
    
    #get id of trajectories with calculated skeletons
    with pd.HDFStore(trajectories_file, 'r') as table_fid:
        df = table_fid['/plate_worms'];
        df = df[['worm_index_joined', 'frame_number', 'coord_x', 'coord_y']]
        df = df.query('worm_index_joined>0')
    #df = df[df[str_index]==4423] 
    #good_index = df[df['segworm_id']>=0]['worm_index_joined'].unique();
    #df = df[df['worm_index_joined'].isin(good_index)];
    
    max_frame = df[['worm_index_joined', 'frame_number']].groupby('worm_index_joined').aggregate(np.max);
    last_frame = max_frame['frame_number'].max()
    #change into a dictionary for speed
    max_frame = max_frame.to_dict()['frame_number']
    
    
    mask_fid = tables.File(masked_image_file, "r");
    I_worms = mask_fid.get_node("/mask")
    
    track_list = {}
    track_colour = {};
    
    movie_save_name = trajectories_file[:-5] + '.avi';
    im_size = tuple([int(x*movie_scale) for x in I_worms.shape[1:]])
    #video_writer = cv2.VideoWriter(movie_save_name, cv2.cv.FOURCC('M','J','P','G'), 25, \
    #                im_size, isColor=True)
    video_writer = writeVideoffmpeg(movie_save_name, im_size[0], im_size[1], pix_fmt = 'rgb24')
    #%%
    progressTime = timeCounterStr('Writing trajectories videos.');
    for frame in range(0, last_frame, n_frames_jumped):
        frame_data = df[df['frame_number'] == frame]
        #clear track that has already ended
        finished_tracks = []
        for worm_ind in track_list.keys():
            if frame > max_frame[worm_ind]:
                finished_tracks.append(worm_ind)
        for worm_ind in finished_tracks:
            track_list.pop(worm_ind)
            track_colour.pop(worm_ind)
        
        for  row_ind, worm_row in frame_data.iterrows():
            worm_ind = worm_row['worm_index_joined']
            
            if not worm_ind in track_list:
                track_list[worm_ind] = [];
                track_colour[worm_ind] = np.random.randint(100, 255, 3) 
                #it seems that cv2.polynes does not accept np.int64
                track_colour[worm_ind] = tuple(map(int, track_colour[worm_ind]))
            track_list[worm_ind].append((worm_row['coord_x'], worm_row['coord_y']))
            if len(track_list[worm_ind]) > max_track_draw:
                track_list[worm_ind].pop(0)
        
        img = I_worms[frame,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB);
    #%%    
        img = cv2.resize(img, (0,0), fx=movie_scale, fy=movie_scale);
        for worm_ind in track_list:
            pts = np.array(list(zip(track_list[worm_ind])))*movie_scale
            cv2.polylines(img, [pts.astype(np.int32)], False, track_colour[worm_ind], thickness=2, lineType = 8) 
        video_writer.write(img)
#    #%%    
        if frame % 1000 == 0:
            progress_str = progressTime.getStr(frame)
            print(base_name + ' ' + progress_str);
#            
    video_writer.release()

if __name__ == '__main__':
#    masked_image_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303.hdf5'
#    trajectories_file = '/Users/ajaver/Desktop/Gecko_compressed/20150512/Capture_Ch3_12052015_194303_trajectories.hdf5'
    masked_image_file = sys.argv[1]
    trajectories_file = sys.argv[2]
    print(masked_image_file, trajectories_file)
    drawTrajectoriesVideo(masked_image_file, trajectories_file)
    