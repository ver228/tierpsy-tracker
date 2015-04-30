# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:46:25 2015

@author: ajaver
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 01:56:06 2015

@author: ajaver
"""
import os
import subprocess as sp
import time
import sys


#def get_tracking_cmd(masked_movies_dir, trajectories_dir, base_name):
#    cmd =  ' '.join(["python trackSingleFile.py", masked_movies_dir, \
#        trajectories_dir, base_name, '</dev/null'])
##    return cmd
#def get_videos_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name):
#    cmd =  ' '.join(["python indVideosSingleFile.py", masked_movies_dir, \
#        trajectories_dir, main_video_save_dir, base_name, \
#        '</dev/null'])
#    return cmd

def get_tracking_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name):
    cmd =  ' '.join(["python trackSingleFile.py", masked_movies_dir, \
        trajectories_dir, main_video_save_dir, base_name, '</dev/null'])
    return cmd
    

#%%

if __name__ == '__main__':
#    masked_movies_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/'
#    trajectories_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/'
#    main_video_save_dir =r'/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/'
#    progress_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/progress_txt/'


#    expDateStr = '20150216'
#    expDateStr = sys.argv[1]
#    masked_movies_dir =  r'/Volumes/behavgenom$/GeckoVideo/Compressed/' + expDateStr + '/'
#    trajectories_dir =  r'/Volumes/behavgenom$/GeckoVideo/Trajectories/' + expDateStr + '/'
#    main_video_save_dir = r'/Volumes/behavgenom$/GeckoVideo/Invidual_videos/'  + expDateStr + '/'
#    progress_dir = r'/Volumes/behavgenom$/GeckoVideo/progress_txt/'
# 

#    masked_movies_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/'
#    trajectories_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/Trajectories/'
#    main_video_save_dir =r'/Users/ajaver/Desktop/Gecko_compressed/20150323/Worm_Movies/'
#    progress_dir = r'/Users/ajaver/Desktop/Gecko_compressed/20150323/progress_txt/'


#    expDateStr = '20150216'
    expDateStr = sys.argv[1]
    masked_movies_dir =  r'/Volumes/behavgenom$/GeckoVideo/Compressed/' + expDateStr + '/'
    trajectories_dir =  r'/Volumes/behavgenom$/GeckoVideo/Trajectories/' + expDateStr + '/'
    main_video_save_dir = r'/Volumes/behavgenom$/GeckoVideo/Invidual_videos/'  + expDateStr + '/'
    progress_dir = r'/Volumes/behavgenom$/GeckoVideo/progress_txt/'
 
#    masked_movies_dir = r'/Volumes/behavgenom$/Alex_Anderson/Compressed/Locomotion_videos_for_analysis_2015-2/'
#    trajectories_dir =  r'/Volumes/behavgenom$/Alex_Anderson/Trajectories/Locomotion_videos_for_analysis_2015-2/'
#    main_video_save_dir = r'/Volumes/behavgenom$/Alex_Anderson/Invidual_videos/Locomotion_videos_for_analysis_2015-2/'
#    progress_dir = r'/Volumes/behavgenom$/Alex_Anderson/progress_txt-2/'

#    masked_movies_dir = r'/Volumes/behavgenom$/Alex_Anderson/Compressed/Locomotion_videos_for_analysis_2015/'
#    trajectories_dir =  r'/Volumes/behavgenom$/Alex_Anderson/Trajectories/Locomotion_videos_for_analysis_2015/'
#    main_video_save_dir = r'/Volumes/behavgenom$/Alex_Anderson/Invidual_videos/Locomotion_videos_for_analysis_2015/'
#    progress_dir = r'/Volumes/behavgenom$/Alex_Anderson/progress_txt/'

#    masked_movies_dir =  '/Volumes/behavgenom$/syngenta/Compressed/data_20150114/'
#    trajectories_dir =  '/Volumes/behavgenom$/syngenta/Trajectories/data_20150114/'
#    main_video_save_dir =  '/Volumes/behavgenom$/syngenta/Invidual_videos/data_20150114/'
#    progress_dir =  '/Volumes/behavgenom$/syngenta/progress_txt/'
    
#%%
    max_num_process = 6;
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)
    if not os.path.exists(main_video_save_dir):
        os.makedirs(main_video_save_dir)
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir)
    
    #get a list 
    file_list = os.listdir(masked_movies_dir);
    base_name_list = [os.path.splitext(x)[0] for x in file_list if ('.hdf5' in x)]#    #start the parallizeTask object, obtain the queue where the progress status is stored
    #base_name_list = base_name_list[-1:]
#%%    
    tot_tasks = len(base_name_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = []        
    progress_dict = {};

    num_tasks = 0; 
    for base_name in base_name_list[0:max_num_process]: 
        progress_file = progress_dir + 'progress_' + base_name + '.txt'
        progress_dict[base_name] = progress_file;

        cmd = get_tracking_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)
                #' >& ' , progress_file, '</dev/null'])
        #& used to redirect both stdout and stderr

        current_tasks.append(sp.Popen(cmd, shell='True'))
        num_tasks += 1
        print('%s : started.' % base_name)

    #when one processs finish start a 
    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
        for ii in range(len(current_tasks)):
            if not current_tasks[ii].poll() is None and num_tasks < tot_tasks:
                base_name = base_name_list[num_tasks]
                
                progress_file = progress_dir + 'progress_' + base_name + '.txt'
                progress_dict[base_name] = progress_file;

                cmd = get_tracking_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)
                #' >& ' , progress_file, '</dev/null'])
                #</dev/null added to avoid annoying msg from MATLAB
                current_tasks[ii] = sp.Popen(cmd, shell='True')
                num_tasks +=1
                print('%s : started.' % base_name)
    
#    max_num_process = 6;
#    num_tasks = 0; 
#    for base_name in base_name_list[0:max_num_process]: 
#        cmd = get_videos_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)
#        #' >& ' , progress_file, '</dev/null'])
#        #& used to redirect both stdout and stderr
#        current_tasks.append(sp.Popen(cmd, shell='True'))
#        num_tasks += 1
#        print('%s : started.' % base_name)
#
#    #when one processs finish start a 
#    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
#        for ii in range(len(current_tasks)):
#            if not current_tasks[ii].poll() is None and num_tasks < tot_tasks:
#                base_name = base_name_list[num_tasks]
#                cmd = cmd = get_videos_cmd(masked_movies_dir, trajectories_dir, main_video_save_dir, base_name)
#                #' >& ' , progress_file, '</dev/null'])
#                #</dev/null added to avoid annoying msg from MATLAB
#                current_tasks[ii] = sp.Popen(cmd, shell='True')
#                num_tasks +=1
#                print('%s : started.' % base_name)
#    
#    print the final status of each file
#    os.system('clear')
#    for base_name in progress_dict:
#       progress_file = progress_dict[base_name]
#       print(base_name)
#       with open(progress_file, 'r') as f:
#            lines = f.readlines()
#            if len(lines)>2:
#                for kk in range(-2,0):            
#                    print(lines[kk][:-1])
#        time.sleep(5)
##        
    