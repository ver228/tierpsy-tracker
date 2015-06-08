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

def get_tracking_cmd(masked_movie_file, results_dir):
    cmd =  ' '.join(["python3 trackSingleFile.py", masked_movie_file, results_dir, '</dev/null'])
    return cmd

def start_task(movie_file, results_dir, current_tasks, num_tasks):
    cmd = get_tracking_cmd(movie_file, results_dir)
    print(cmd)
    current_tasks.append(sp.Popen(cmd, shell='True'))
    num_tasks += 1
    print('%s : started.' % movie_file)
    return current_tasks, num_tasks
    

#%%

if __name__ == '__main__':
    
    masked_movies_dir = '/Users/ajaver/Desktop/Gecko_compressed/Masked_Videos/20150512/'
    results_dir = '/Users/ajaver/Desktop/Gecko_compressed/Results/20150512/'
    
#%%
    max_num_process = 6;
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #get a file list 
    masked_movies_list = os.listdir(masked_movies_dir);
    #filter for files with .hdf5 and add the full path
    masked_movies_list = [masked_movies_dir + x for x in masked_movies_list if ('.hdf5' in x)]
    
    tot_tasks = len(masked_movies_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = []        
    progress_dict = {};

    num_tasks = 0; 
    for movie_file in masked_movies_list[0:max_num_process]:
        current_tasks, num_tasks = start_task(movie_file, results_dir, current_tasks, num_tasks)

    #when one processs finish start a new one 
    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
        for ii in range(len(current_tasks)):
            if not current_tasks[ii].poll() is None and num_tasks < tot_tasks:
                current_tasks, num_tasks = start_task(movie_file, results_dir, current_tasks, num_tasks)
    
