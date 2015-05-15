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
import subprocess as sp
import glob

if __name__ == '__main__':
    rootDir = r'/Volumes/behavgenom$/GeckoVideo/Compressed/'
    main_cmd = 'python correct_im_diff '
    max_num_process = 6
#%%    
    #get a list 
    
    base_name_list = glob.glob(rootDir + '*/*.hdf5')#    #start the parallizeTask object, obtain the queue where the progress status is stored

    
    tot_tasks = len(base_name_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = []        
    progress_dict = {};

    num_tasks = 0; 
    for base_name in base_name_list[0:max_num_process]: 
        cmd = main_cmd + base_name


        current_tasks.append(sp.Popen(cmd, shell='True'))
        num_tasks += 1
        print('%s : started.' % base_name)

    #when one processs finish start a 
    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
        if len(current_tasks) == 0:
            break;
        for ii in range(len(current_tasks)):
            if not current_tasks[ii].poll() is None and num_tasks < tot_tasks:
                base_name = base_name_list[num_tasks]
                
                cmd = main_cmd + base_name

                current_tasks[ii] = sp.Popen(cmd, shell='True')
                num_tasks +=1
                print('%s : started.' % base_name)
    
