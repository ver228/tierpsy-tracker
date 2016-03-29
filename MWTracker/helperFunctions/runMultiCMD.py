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


from io import StringIO
class CapturingOutput(list):
    '''modified from http://stackoverflow.com/questions/1218933/can-i-redirect-the-stdout-in-python-into-some-sort-of-string-buffer'''
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend([x + '\n' for x in self._stringio.getvalue().splitlines()])
        sys.stdout = self._stdout

ON_POSIX = 'posix' in sys.builtin_module_names
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

class start_process():
    def __init__(self, cmd, local_obj=''):
        self.output = ['Started\n']
            
        if local_obj:
            with CapturingOutput() as output:
                self.obj_cmd = local_obj(cmd[1:])
                self.cmd = self.obj_cmd.start()
            self.output += output
            
            #print(self.cmd, type(self.cmd))
            #import pdb
            #pdb.set_trace()
            #print(output[-1])    

        else:
            self.obj_cmd = ''
            self.cmd = cmd
            self.output = ['Started\n']
        
        self.pid = sp.Popen(self.cmd, stdout = sp.PIPE, stderr = sp.PIPE,
                            bufsize = 1, close_fds = ON_POSIX)
        self.queue = Queue()
        self.thread = Thread(target = enqueue_output, 
                             args = (self.pid.stdout, self.queue))
        self.thread.start()

    def read_buff(self):
        while 1:    
            # read line without blocking
            try: self.output.append(self.queue.get_nowait().decode("utf-8"))
            except Empty: break
        #store only the last line
        self.output = self.output[-1:]

    def close(self):
        
        if self.pid.poll() != 0:
            #print errors details if there was any
            self.output[-1] += 'ERROR: \n'

            self.output[-1] += cmdlist2str(self.cmd) + '\n'
            self.output[-1] += self.pid.stderr.read().decode("utf-8")
            self.pid.stderr.flush()
        
        if self.obj_cmd and self.pid.poll() == 0:
            with CapturingOutput() as output:
                self.obj_cmd.clean()
            self.output += output
            
        self.pid.wait()
        self.pid.stdout.close()
        self.pid.stderr.close()

def runMultiCMD(cmd_list, local_obj='', max_num_process = 3, refresh_time = 10):
    '''Start different process using the command is cmd_list'''
    cmd_list = cmd_list[::-1] #since I am using pop to get the next element i need to invert the list to get athe same order
    tot_tasks = len(cmd_list)
    if tot_tasks < max_num_process:
        max_num_process = tot_tasks
    
    #initialize the first max_number_process in the list
    finished_tasks = [];
    num_tasks = 0;
    
    
    current_tasks = [];
    for ii in range(max_num_process):
        cmd = cmd_list.pop()
        current_tasks.append(start_process(cmd, local_obj))
    
    #keep loop tasks as long as there is any task alive and 
    #the number of tasks stated is less than the total number of tasks
    while cmd_list or any(tasks.pid.poll() is None for tasks in current_tasks):
        time.sleep(refresh_time)
        os.system(['clear','cls'][os.name == 'nt'])
        
        #print info of the finished tasks
        for task in finished_tasks: 
            sys.stdout.write(task)

        #loop along the process list to update output and see if there is any task finished
        next_tasks = []
        for task in current_tasks:
            task.read_buff()
            if task.pid.poll() is None:
                #add task to the new list if it hasn't complete
                next_tasks.append(task)
                sys.stdout.write(task.output[-1])
            else:
                #close the task and add its las output to the finished_tasks list
                task.close()
                sys.stdout.write(task.output[-1])
                finished_tasks.append(task.output[-1])
                #add new task once the previous one was finished
                if cmd_list and len(next_tasks) < max_num_process:
                    cmd = cmd_list.pop()            
                    next_tasks.append(start_process(cmd,local_obj))
            
            
        #if there is stlll space add a new tasks.
        while cmd_list and len(next_tasks) < max_num_process:
            cmd = cmd_list.pop()
            next_tasks.append(start_process(cmd, local_obj))
        
        current_tasks = next_tasks

        print('*************************************************')
        print('Tasks: %i finished, %i remaining.' % (len(finished_tasks), len(current_tasks) + len(cmd_list)))
        print('*************************************************')

def cmdlist2str(cmdlist):
    #change the format from the list accepted by Popen to a text string accepted by the terminal 
    for ii, dd in enumerate(cmdlist):
        if ii >= 2 and not dd.startswith('-'):
            dd = "'" + dd + "'"

        if ii == 0:
            cmd_str = dd
        else:
            cmd_str += ' ' +  dd
    return cmd_str

def print_cmd_list(cmd_list_compress):
    #print all the commands to be processed 
    if cmd_list_compress: 
        for cmd in cmd_list_compress: 
            cmd_str = cmdlist2str(cmd)
            print(cmd_str)

    print(len(cmd_list_compress))


