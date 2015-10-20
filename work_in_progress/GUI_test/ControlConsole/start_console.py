# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:43:30 2015

@author: ajaver
"""

#import os

import sys, os
import time
import subprocess as sp
from threading  import Thread
from queue import Queue, Empty


ON_POSIX = 'posix' in sys.builtin_module_names
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

class start_process():
    def __init__(self, cmd):
        self.cmd = cmd
        
        self.pid = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE,
                            bufsize = 1, close_fds = ON_POSIX)
        self.output = ['Started\n']
        self.queue = Queue()
        self.thread = Thread(target = enqueue_output, 
                             args = (self.pid.stdout, self.queue))
        self.thread.start()

    def read_buff(self):
        while 1:    
            # read line without blocking
            try: self.output.append(self.queue.get_nowait().decode("utf-8"))
            except Empty: break

def runMultiCMD(cmd_list, max_num_process = 3, refresh_time = 10):
    '''Start different process using the command is cmd_list'''
    tot_tasks = len(cmd_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    current_tasks = [];
    num_tasks = 0;
    
    
    for ii in range(max_num_process):
        cmd = cmd_list.pop()
        current_tasks.append(start_process(cmd))
    
    #keep loop tasks as long as there is any task alive and 
    #the number of tasks stated is less than the total number of tasks
    while cmd_list or any(tasks.pid.poll() is None for tasks in current_tasks):

        time.sleep(refresh_time)
        os.system(['clear','cls'][os.name == 'nt'])
        #loop along the process list to see if there is a task finished
        for task in current_tasks:
            if task is not None:        
                task.read_buff()
                if task.output:
                    last_line = task.output[-1]
                    sys.stdout.write(last_line)
            
            if task.pid.poll() is not None:
                #check if the current task finished
                if task.pid.poll() != 0:
                    sys.stdout.write(task.pid.stderr.read().decode("utf-8"))
                
                #task = None
                
                if cmd_list:
                    cmd = cmd_list.pop()
                    current_tasks.append(start_process(cmd))
        print('%%%%%%%%%%')


#%%
#cmd_list = []
#for kk in range(1, 10):
#    cmd_list += [['python3', 'dum_count.py', str(kk)]]

#%%

