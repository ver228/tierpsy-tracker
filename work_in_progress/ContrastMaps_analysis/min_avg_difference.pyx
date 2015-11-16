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


cdef inline DTYPE_t absDiff(DTYPE_t a, DTYPE_t b): 
    return a-b if a>b else b-a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def min_avg_difference(np.ndarray[unsigned int, ndim=3] buff_prev, np.ndarray[unsigned int, ndim=3] buff_next):
    cdef DTYPE_t MAX_INT = 2**63 - 1
    cdef int n_row = buff_prev.shape[1];
    cdef int n_col = buff_prev.shape[2];
    cdef int n_prev = buff_prev.shape[0];
    cdef int n_curr = buff_next.shape[0];
    
    
    cdef int i, j, kp, kn
    cdef int current_ind;
    cdef DTYPE_t total, current_min;
    
    cdef np.ndarray[DTYPE_t, ndim=1] min_avg = np.empty(n_curr, dtype= np.int);
    cdef np.ndarray[DTYPE_t, ndim=1] min_avg_index = np.empty(n_curr, dtype= np.int);
    
    for kn in range(n_curr):
        current_min = MAX_INT
        current_ind = -1;
        for kp in range(n_prev):
            total = 0            
            for i in range(n_row):
                for j in range(n_col):
                    if total > current_min:
                        break;
                    total += absDiff(buff_prev[kp,i,j], buff_next[kn,i,j])
            if total < current_min:
                current_min = total;
                current_ind = kp;
                
        min_avg[kn] = total;
        min_avg_index[kn] = current_ind
    
    #return min_avg
    return (min_avg, min_avg_index); 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def min_avg_difference2(np.ndarray[unsigned int, ndim=3] buff_prev, np.ndarray[unsigned int, ndim=3] buff_next):
    cdef DTYPE_t MAX_INT = 2**63 - 1
    cdef int n_row = buff_prev.shape[1];
    cdef int n_col = buff_prev.shape[2];
    cdef int n_prev = buff_prev.shape[0];
    cdef int n_curr = buff_next.shape[0];
    
    
    cdef int i, j, kp, kn
    cdef int current_ind;
    cdef float total, current_min;
    
    cdef np.ndarray[np.float_t, ndim=1] min_avg = np.zeros(n_curr, dtype= np.float);
    cdef np.ndarray[np.int_t, ndim=1] min_avg_index = np.empty(n_curr, dtype= np.int);
    
    cdef np.ndarray[np.float_t, ndim=1] tot_pix_curr = np.zeros(n_prev, dtype= np.float);
    cdef np.ndarray[np.float_t, ndim=1] tot_pix_prev = np.zeros(n_curr, dtype= np.float);


    for kn in range(n_curr):
        tot_pix_curr[kn] = <float>np.sum(buff_next[kn,:,:])

    for kn in range(n_prev):
        tot_pix_prev[kp] = <float>np.sum(buff_prev[kp,:,:])
        

    for kn in range(n_curr):
        current_min = MAX_INT
        current_ind = -1;
        for kp in range(n_prev):
            total = 0            
            for i in range(n_row):
                for j in range(n_col):
                    if total > current_min:
                        break;
                    total += absDiff(buff_prev[kp,i,j], buff_next[kn,i,j])/tot_pix_curr[kn]
            if total < current_min:
                current_min = total;
                current_ind = kp;
                
        min_avg[kn] = total;
        min_avg_index[kn] = current_ind
    
    #return min_avg
    return (min_avg, min_avg_index); 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def avg_difference_mat(np.ndarray[unsigned int, ndim=3] buff_prev, np.ndarray[unsigned int, ndim=3] buff_next):
    cdef DTYPE_t MAX_INT = 2**63 - 1
    cdef int n_row = buff_prev.shape[1];
    cdef int n_col = buff_prev.shape[2];
    cdef int n_prev = buff_prev.shape[0];
    cdef int n_curr = buff_next.shape[0];
    
    
    cdef int i, j, kp, kn
    #cdef int current_ind;
    cdef DTYPE_t total, current_min;
    #cdef np.ndarray[DTYPE_t, ndim=2] h = np.zeros([n_prev, n_curr], dtype = np.int);
    
    cdef np.ndarray[DTYPE_t, ndim=2] diff_avg = np.empty((n_prev,n_curr), dtype= np.int);
    #cdef np.ndarray[DTYPE_t, ndim=1] min_avg_index = np.empty(n_curr, dtype= np.int);
    
    for kn in range(n_curr):
        for kp in range(n_prev):
            total = 0            
            for i in range(n_row):
                for j in range(n_col):
                    total += absDiff(buff_prev[kp,i,j], buff_next[kn,i,j])
            diff_avg[kp, kn] = total;
    return diff_avg

