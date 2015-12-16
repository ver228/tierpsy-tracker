# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:43:30 2015

@author: ajaver
"""
import sys
from start_console import runMultiCMD

if __name__ == "__main__":

	cmd_list = [['python3', 'test_echo.py', str(ii)] for ii in range(int(sys.argv[1]))] 
	print(cmd_list)

	runMultiCMD(cmd_list, max_num_process = 6, refresh_time = 1)