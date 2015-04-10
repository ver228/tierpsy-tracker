# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:31:52 2015

@author: ajaver
"""

while True:#any(tasks.poll()==None for tasks in current_tasks):
   os.system('clear')
   for base_name in progress_dict:
       progress_file = progress_dict[base_name]
       print(base_name)
       with open(progress_file, 'r') as f:
            lines = f.readlines()
            for kk in range(-2,0):            
                print(lines[kk][:-1])
   time.sleep(1)