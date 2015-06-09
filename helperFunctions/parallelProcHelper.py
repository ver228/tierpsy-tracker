# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:27:18 2015

@author: ajaver
"""

import sys
import collections
import subprocess as sp

def startTask(cmd_str, current_tasks, proc_ind, num_tasks):
    print(cmd_str)
    current_tasks[proc_ind] = sp.Popen(cmd_str, shell='True')
    num_tasks += 1
    return current_tasks, num_tasks

def runMultiSubproc(cmd_list, max_num_process = 6):
    '''Start different process using the command is cmd_list'''
    tot_tasks = len(cmd_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = ['']*max_num_process        
    
    num_tasks = 0; 
    for proc_ind, cmd in enumerate(cmd_list[0:max_num_process]):
        current_tasks,num_tasks = startTask(cmd, current_tasks, proc_ind, num_tasks)
    
    
    #keep loop tasks as long as there is any task alive and 
    #the number of tasks stated is less than the total number of tasks
    while num_tasks < tot_tasks or any(tasks.poll()==None for tasks in current_tasks):
        #loop along the process list to see if there is a task finished
        for proc_ind in range(len(current_tasks)):
            
            #start a new tasks if the taks is finished (it is not none) and there is still tasks to add
            if current_tasks[proc_ind].poll() != None and num_tasks < tot_tasks:
                cmd = cmd_list[num_tasks]
                current_tasks, num_tasks = startTask(cmd, current_tasks, proc_ind, num_tasks)


#The next code uses the multiprocess module. This module can be problematic. When it works
#it works nicely, but in some systems and for some libraries (opencv) it crashes the 
#program
import multiprocessing as mp

def sendQueueOrPrint(status_queue, progress_str, base_name):
    '''small code to decide if the progress must be send to a queue or printed in the screen'''
    if type(status_queue).__name__ == 'Queue':
        status_queue.put([base_name, progress_str]) 
    else:
        print ('%s:\t %s' % (base_name, progress_str))

def printProgress(progress):
    '''useful function to write the progress status in multiprocesses mode'''
    sys.stdout.write('\033[2J')
    sys.stdout.write('\033[H')
    for filename, progress_str in progress.items():
        print(filename + ' ' + progress_str)

    sys.stdout.flush()
    
class parallelizeTask:
    '''using the multiprocess module'''
    def __init__(self, max_num_process = 6):
        self.status_queue = mp.Queue()
        self.max_num_process = max_num_process;
        
    def start(self, workers_function, workers_arg):
        if len(workers_arg) < self.max_num_process:
            self.max_num_process = len(workers_arg)
        
        workers = []
        progress = collections.OrderedDict()
        for base_name in workers_arg:
            #start a process for each video file in save_dir
            child = mp.Process(target = workers_function, args=workers_arg[base_name])
            workers.append(child)
            progress[base_name] = 'idle'
        
        current_workers_id = range(self.max_num_process)
        remaining_workers_id = range(self.max_num_process, len(workers))
         
        for worker_id in current_workers_id:
            workers[worker_id].start()
        
        while remaining_workers_id or \
            any(workers[worker_id].is_alive() for worker_id in current_workers_id):
            
            #add a new worker if one has already finished
            for ii, worker_id in enumerate(current_workers_id):
                if not workers[worker_id].is_alive() and remaining_workers_id:
                    new_id = remaining_workers_id.pop()                
                    workers[new_id].start()
                    current_workers_id[ii] = new_id
            
            time.sleep(1.0) # I made this value larger because I can only save the output in a file on a schedule task with "at". Refreshing it too frequently will produce a huge file.
            while not self.status_queue.empty():
                filename, percent = self.status_queue.get()
                progress[filename] = percent
                printProgress(progress)