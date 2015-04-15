# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:27:18 2015

@author: ajaver
"""

import sys
import collections
import multiprocessing as mp
import time, datetime

class timeCounterStr:
    def __init__(self, task_str):
        self.initial_time = time.time();
        self.last_frame = 0;
        self.task_str = task_str;
        self.fps_time = float('nan');
        
    def getStr(self, frame_number):
        #calculate the progress and put it in a string
        time_str = str(datetime.timedelta(seconds = round(time.time()-self.initial_time)))
        fps = (frame_number-self.last_frame+1)/(time.time()-self.fps_time)
        progress_str = '%s Total time = %s, fps = %2.1f; Frame %i '\
            % (self.task_str, time_str, fps, frame_number)
        self.fps_time = time.time()
        self.last_frame = frame_number;
        return progress_str;
        
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
        print filename, progress_str

    sys.stdout.flush()
    
class parallelizeTask:
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