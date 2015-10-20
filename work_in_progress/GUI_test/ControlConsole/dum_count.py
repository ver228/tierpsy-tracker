# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 23:33:37 2015

@author: ajaver
"""

import sys
import time

if __name__ == '__main__':
    N = int(sys.argv[1])
    
    for n in range(N):
        print('%i Count %i' % (N, n+1))
        sys.stdout.flush()
        time.sleep(1)
    print('%i Exit' % N)
    #raise