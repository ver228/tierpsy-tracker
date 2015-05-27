# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:12:17 2015

@author: ajaver
"""

import glob
import subprocess as sp

if __name__ == '__main__':
    root_dir = '/Volumes/behavgenom$/GeckoVideo/Trajectories/'
    main_cmd = 'python getDrawTrajectories.py'
    max_num_process = 12
#%%    
    #get a list 
    
    base_name_list = glob.glob(root_dir + '*/*_trajectories.hdf5')#    #start the parallizeTask object, obtain the queue where the progress status is stored

    
    tot_tasks = len(base_name_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = []        
    progress_dict = {};

    num_tasks = 0; 
    for trajectories_file in base_name_list[0:max_num_process]: 
        masked_image_file = trajectories_file.replace('Trajectories', 'Compressed').replace('_trajectories', '')
        cmd = ' '.join([main_cmd, masked_image_file, trajectories_file]);


        current_tasks.append(sp.Popen(cmd, shell='True'))
        num_tasks += 1
        #print(cmd)

    #when one processs finish start a 
    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
        if len(current_tasks) == 0:
            break;
        for ii in range(len(current_tasks)):
            if not current_tasks[ii].poll() is None and num_tasks < tot_tasks:
                base_name = base_name_list[num_tasks]
                
                cmd = ' '.join([main_cmd, masked_image_file, trajectories_file]);

                current_tasks[ii] = sp.Popen(cmd, shell='True')
                num_tasks +=1
                #print(cmd)