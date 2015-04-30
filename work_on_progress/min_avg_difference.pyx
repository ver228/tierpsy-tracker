# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 17:54:42 2015

@author: ajaver
"""
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs as c_abs

ctypedef np.int64_t DTYPE_t
from numpy.math cimport INFINITY

@cython.boundscheck(False)
cdef inline DTYPE_t absDiff(DTYPE_t a, DTYPE_t b): 
    return a-b if a>b else b-a

def min_avg_difference(np.ndarray[unsigned int, ndim=3] buff_prev, np.ndarray[unsigned int, ndim=3] buff_next):
    #f, g are to one dim vector
    cdef DTYPE_t MAX_INT = 2**63 - 1
    cdef int n_row = buff_prev.shape[1];
    cdef int n_col = buff_prev.shape[2];
    cdef int n_prev = buff_prev.shape[0];
    cdef int n_curr = buff_next.shape[0];
    
    
    cdef int i, j, kp, kn;
    cdef DTYPE_t total, current_min;
    #cdef double tot_pix = 0;
    #cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([n_prev, n_curr], dtype = np.int);
    
    cdef np.ndarray[DTYPE_t, ndim=1] min_avg = np.empty(n_curr, dtype= np.int);
    
    for kn in range(n_curr):
        current_min = MAX_INT
        for kp in range(n_prev):
            total = 0            
            for i in range(n_row):
                for j in range(n_col):
                    if total > current_min:
                        break;
                    total += absDiff(buff_prev[kp,i,j], buff_next[kn,i,j])
            current_min = min(current_min, total)
            
            
        min_avg[kn] = total;
    
    #print tot_pix
    return min_avg; #total/<double>f.size;
    